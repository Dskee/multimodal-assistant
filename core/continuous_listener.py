# -*- coding: utf-8 -*-
"""
连续语音输入设备层：麦克风 + WASAPI 回环 双流采集

core/audio_input.py 里是"算法"（已在合成信号上自检通过），本模块负责把它接到真机上。
分开写的原因：算法可以用合成信号严格验证，设备层只能靠真机；混在一起会让两者都难测。

【两条流】
    麦克风   16kHz 单声道 —— 用户说的话 + 扬声器的回声
    回环     设备默认采样率（常见 48kHz 立体声）—— "系统正在播放什么"，即回声的参考信号
    回环那条必须重采样成 16kHz 单声道，才能和麦克风逐样本对齐。

【三个容易搞错的地方，都在代码里写清了原因】
    1. 参考信号要"延后" delay 个采样再与麦克风配对（回声是 d 个采样之前播出来的），
       方向搞反会让等效时延变成 2d，回声消除效果腰斩。
    2. VAD 要接在**回声消除后的残差**上，而不是原始麦克风——
       否则程序会把自己的播报切成一"句话"再送去识别，形成自问自答。
    3. 回环流的块长与麦克风不同（48k/16k = 3 倍），必须按样本数换算，
       不能想当然地用同一个 frames_per_buffer。

【隐私】本模块只做电平与事件计算，默认不落盘任何音频。probe() 只读电平、不保存。
"""

import os
import sys
import threading
import time

import numpy as np

from config import (
    RECORD_RATE, RECORD_CHANNELS, RECORD_DEVICE,
    VAD_FRAME_MS, AEC_ENABLED, AEC_REF_DEVICE_INDEX, AEC_FILTER_MS, AEC_STEP_SIZE,
    BARGE_IN_ENABLED,
)
from core.audio_input import (
    VadSegmenter, EchoCanceller, BargeInDetector,
    pcm_to_float, float_to_pcm, rms, list_loopback_devices, db_from_ratio,
)


# ======================================================================
# 重采样：回环(常见 48kHz 立体声) -> 16kHz 单声道
# ======================================================================

class Resampler:
    """
    流式重采样 + 声道降混。

    跨块连续性靠两点保证：
      - 抗混叠用的移动平均保留尾部 self._ma_len-1 个样本
      - 重采样位置用"剩余缓冲区 + 相位"记账，不用块内独立下标
        （否则每块都会重新从 0 开始，产生周期性抖动）
    """

    def __init__(self, in_rate, out_rate, channels=1):
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.channels = max(1, int(channels))
        self.ratio = float(self.in_rate) / float(self.out_rate)
        self._ma_len = max(1, int(round(self.ratio)))
        self._tail = np.zeros(self._ma_len - 1, dtype=np.float32) if self._ma_len > 1 else None
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0.0

    def process(self, pcm_bytes):
        """int16 PCM -> float32 16kHz 单声道"""
        if not pcm_bytes:
            return np.zeros(0, dtype=np.float32)
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1:
            n = (len(x) // self.channels) * self.channels
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            x = x[:n].reshape(-1, self.channels).mean(axis=1)

        # 抗混叠：整数倍抽取用移动平均已足够（48k→16k 正好 3 倍）
        if self._ma_len > 1:
            ext = np.concatenate([self._tail, x])
            k = np.ones(self._ma_len, dtype=np.float32) / self._ma_len
            x = np.convolve(ext, k, mode="valid").astype(np.float32)
            self._tail = ext[-(self._ma_len - 1):]

        if abs(self.ratio - 1.0) < 1e-9:
            return x

        self._buf = np.concatenate([self._buf, x])
        out = []
        while self._pos + 1.0 < len(self._buf):
            i = int(self._pos)
            frac = self._pos - i
            out.append(self._buf[i] * (1.0 - frac) + self._buf[i + 1] * frac)
            self._pos += self.ratio
        consumed = int(self._pos)
        if consumed > 0:
            self._buf = self._buf[consumed:]
            self._pos -= consumed
        return np.asarray(out, dtype=np.float32)


# ======================================================================
# 连续监听
# ======================================================================

class ContinuousListener:
    """
    连续语音输入。

    典型用法（界面线程里）：
        lis = ContinuousListener(on_utterance=..., on_barge_in=..., on_status=...)
        lis.start()                       # 开始一直听
        lis.set_playing(True)             # 开始播报：进入打断检测 + 自动标定回声路径
        lis.set_playing(False)            # 播报结束：回到聆听
        lis.stop()

    回调都在采集线程里触发，界面层负责切回自己的线程（Qt 用信号）。
    """

    def __init__(self, on_utterance=None, on_barge_in=None, on_status=None,
                 sample_rate=RECORD_RATE, block_ms=VAD_FRAME_MS,
                 mic_device=RECORD_DEVICE, loopback_device=AEC_REF_DEVICE_INDEX,
                 enable_aec=AEC_ENABLED, enable_barge_in=BARGE_IN_ENABLED):
        self.on_utterance = on_utterance
        self.on_barge_in = on_barge_in
        self.on_status = on_status

        self.sample_rate = int(sample_rate)
        self.block_ms = int(block_ms)
        self.block_samples = int(self.sample_rate * self.block_ms / 1000.0)
        self.mic_device = mic_device
        self.loopback_device = loopback_device
        self.enable_aec = bool(enable_aec)
        self.enable_barge_in = bool(enable_barge_in)

        # 算法部件
        self.segmenter = VadSegmenter(sample_rate=self.sample_rate, frame_ms=self.block_ms)
        self.aec = EchoCanceller(sample_rate=self.sample_rate,
                                 filter_ms=AEC_FILTER_MS, step_size=AEC_STEP_SIZE) \
            if self.enable_aec else None
        self.barge_in = BargeInDetector(sample_rate=self.sample_rate) \
            if self.enable_barge_in else None

        # 运行状态
        self._stop = threading.Event()
        self._thread = None
        self._playing = False
        self._playing_since = 0.0
        self._mic_stream = None
        self._ref_stream = None
        self._pa = None
        self._pa_wp = None
        self._resampler = None
        self._ref_ratio = 1.0

        # 缓冲与统计
        self._ref_buf = np.zeros(0, dtype=np.float32)
        self._ref_delay_line = np.zeros(0, dtype=np.float32)   # 参考信号延迟线
        self._calib_mic = []
        self._calib_ref = []
        self._calibrating = False
        self._calib_need = 0
        self._calib_pending = False
        self.mic_channels = 1                # 打开成功后被 _open_mic 覆盖
        self.mic_device_used = None
        self.loopback_info = None
        self._echo_control_ok = False        # 没有回环就没有回声消除，也没有打断

        # 回环必须由**独立线程**读取并放进队列：
        # WASAPI 回环在没有音频播放时 read() 会一直阻塞，
        # 如果放在主循环里读，没放声音时整个采集循环会被卡死，
        # 表现就是"播报一结束，连续对话就不动了"（探针实测到 4.3 秒只回一个块）。
        self._ref_queue = np.zeros(0, dtype=np.float32)
        self._ref_lock = threading.Lock()
        self._ref_thread = None
        self._ref_abort = False
        self._ref_error = ""

        self.error = ""
        self.stats = {
            "blocks": 0, "utterances": 0, "barge_ins": 0,
            "level": 0.0, "residual_level": 0.0, "erle_db": 0.0,
            "ref_padding": 0, "started_at": 0.0,
        }

    # -------------------- 生命周期 --------------------

    def start(self):
        """打开设备并启动采集线程。返回 (是否成功, 错误信息)"""
        if self._thread and self._thread.is_alive():
            return True, ""
        import pyaudio
        try:
            self._pa = pyaudio.PyAudio()
            self._mic_stream = self._open_mic(self._pa)
        except Exception as e:
            self.error = "麦克风打开失败: %s" % e
            self._cleanup()
            return False, self.error

        # 回环是"可选增强"。注意：拿不到回环就必须**同时关掉打断检测**——
        # 没有回声消除时，残差里全是自己的播报声，打断会疯狂误触发。
        self._echo_control_ok = False
        if self.enable_aec:
            try:
                import pyaudiowpatch as pyaudio_wp
                self._pa_wp = pyaudio_wp.PyAudio()
                # 回环用回调模式打开；重采样器在 _open_loopback 内部创建
                self._ref_stream, self._ref_ratio = self._open_loopback(self._pa_wp)
                self._echo_control_ok = True
            except Exception as e:
                self._ref_stream = None
                self.enable_barge_in = False
                self.barge_in = None
                self._emit_status("回环设备不可用（%s）：本次不启用回声消除与打断检测。"
                                  "播报期间请勿说话，或用按钮打断。" % str(e)[:60])

        self._stop.clear()
        self._ref_abort = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.stats["started_at"] = time.time()
        return True, ""

    def stop(self, timeout=2.0):
        # 先让回环回调主动退出（返回 paComplete），再关流。
        # 顺序很重要：回调模式下没有线程阻塞在 read()，因此不会出现"边关边读"的访问冲突。
        self._ref_abort = True
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._cleanup()

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def get_level(self):
        """当前麦克风电平（给界面画音量条用）"""
        return float(self.stats.get("level", 0.0))

    def set_playing(self, playing):
        """
        告诉监听器"现在开始/结束播报"。
        开始播报时会进入回声路径标定（前 calib_ms 毫秒），标定期间不判定打断。
        """
        playing = bool(playing)
        if playing == self._playing:
            return
        self._playing = playing
        self._playing_since = time.time()
        if playing:
            self.segmenter.reset()
            self._ref_delay_line = np.zeros(0, dtype=np.float32)
            if self.barge_in:
                self.barge_in.reset()
            if self.aec is not None and not self.aec.calibrated:
                # 只是"排队等标定"：等回环真的送来声音再开始，
                # 否则会拿一堆静音去估时延，得到相关系数≈0 的垃圾参数
                self._calib_pending = True
        else:
            self._calibrating = False
            self._calib_pending = False
            self._calib_mic = []
            self._calib_ref = []

    # -------------------- 设备 --------------------

    def _open_mic(self, pa):
        """按 设备/声道 组合逐个尝试，与 core/recorder.py 的降级策略一致"""
        import pyaudio
        candidates = []
        if self.mic_device is not None:
            candidates.append((RECORD_CHANNELS, self.mic_device))
            candidates.append((1, self.mic_device))
        candidates.append((RECORD_CHANNELS, None))
        candidates.append((1, None))
        seen, ordered = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        last_err = None
        for channels, device in ordered:
            try:
                kwargs = {"input_device_index": device} if device is not None else {}
                stream = pa.open(format=pyaudio.paInt16, channels=channels,
                                 rate=self.sample_rate, input=True,
                                 frames_per_buffer=self.block_samples, **kwargs)
                self.mic_channels = channels
                self.mic_device_used = device
                return stream
            except Exception as e:
                last_err = e
                continue
        raise Exception(last_err or "没有可用输入设备")

    def _open_loopback(self, pa_wp):
        """
        打开回环流，返回 (stream, in_rate/out_rate)。

        【为什么用回调模式而不是 read()】
        WASAPI 回环在没有音频播放时 read() 会一直阻塞。如果用一个线程去 read()，
        关闭时 join 必然超时，主线程随后 close() 那个流，而读线程还在里面 ——
        实测直接触发访问冲突（0xC0000005）把整个进程干掉。
        回调模式下没有 Python 线程阻塞在 read() 上，关闭就是安全的；
        而且回调不来数据恰好等价于"当前没在播报"，语义正确。
        """
        if self.loopback_device is not None:
            info = pa_wp.get_device_info_by_index(int(self.loopback_device))
        else:
            devs = list_loopback_devices()
            if not devs or "error" in devs[0]:
                raise Exception("未找到回环设备")
            info = pa_wp.get_device_info_by_index(int(devs[0]["index"]))

        in_rate = int(info.get("defaultSampleRate", 48000))
        channels = int(info.get("maxInputChannels", 2)) or 2
        ratio = in_rate / float(self.sample_rate)
        self._ref_ratio = ratio
        # 重采样器必须先建好：回调里要用
        self._resampler = Resampler(in_rate=in_rate, out_rate=self.sample_rate,
                                    channels=channels)
        frames = max(64, int(round(self.block_samples * ratio)))

        def _callback(in_data, frame_count, time_info, status):
            if self._ref_abort or self._stop.is_set():
                return None, pa_wp.paComplete
            try:
                y = self._resampler.process(in_data)
            except Exception:
                return None, pa_wp.paContinue
            if y is not None and len(y):
                with self._ref_lock:
                    self._ref_queue = np.concatenate([self._ref_queue, y])
                    cap = self.sample_rate // 2          # 积压上限 0.5 秒
                    if len(self._ref_queue) > cap:
                        self._ref_queue = self._ref_queue[-cap:]
                    self.stats["ref_blocks"] = self.stats.get("ref_blocks", 0) + 1
            return None, pa_wp.paContinue

        stream = pa_wp.open(format=8,   # paInt16
                            channels=channels, rate=in_rate, input=True,
                            input_device_index=int(info["index"]),
                            frames_per_buffer=frames,
                            stream_callback=_callback)
        stream.start_stream()
        self.loopback_info = {"index": int(info["index"]), "name": str(info.get("name", "")),
                              "rate": in_rate, "channels": channels}
        return stream, ratio

    def _cleanup(self):
        # 先置放弃标志，让回环回调返回 paComplete 自行退出
        self._ref_abort = True
        for s in (self._mic_stream, self._ref_stream):
            try:
                if s is not None:
                    s.stop_stream()
            except Exception:
                pass
        for s in (self._mic_stream, self._ref_stream):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        self._mic_stream = None
        self._ref_stream = None
        for pa in (self._pa, self._pa_wp):
            try:
                if pa is not None:
                    pa.terminate()
            except Exception:
                pass
        self._pa = None
        self._pa_wp = None

    # -------------------- 标定 --------------------

    def _start_calibration(self, calib_ms=600):
        self._calibrating = True
        self._calib_need = int(self.sample_rate * calib_ms / 1000.0)
        self._calib_mic = []
        self._calib_ref = []

    def _feed_calibration(self, mic, ref):
        """收集一段"只有回声"的信号，然后估计时延与增益"""
        self._calib_mic.append(mic)
        self._calib_ref.append(ref)
        if sum(len(x) for x in self._calib_mic) < self._calib_need:
            return
        mic_cat = np.concatenate(self._calib_mic)
        ref_cat = np.concatenate(self._calib_ref)
        self._calib_mic = []
        self._calib_ref = []
        self._calibrating = False
        delay, gain, corr = self.aec.calibrate(mic_cat, ref_cat)
        # 相关度太低说明这段里有人说话或根本没在播报，别用这组参数
        if corr < 0.4:
            self.aec.calibrated = False
            self._emit_status("回声路径标定不可靠（相关度 %.2f），下次播报再试" % corr)
        else:
            self._emit_status("回声路径已标定：时延 %.0fms，增益 %.2f，相关度 %.2f"
                              % (delay / self.sample_rate * 1000, gain, corr))

    # -------------------- 采集主循环 --------------------

    def _loop(self):
        blocks = 0
        try:
            while not self._stop.is_set():
                # 1) 麦克风
                try:
                    mic_bytes = self._mic_stream.read(self.block_samples,
                                                      exception_on_overflow=False)
                except Exception as e:
                    self.error = "麦克风读取失败: %s" % e
                    break
                mic = pcm_to_float(mic_bytes)
                if self.mic_channels > 1:
                    n = (len(mic) // self.mic_channels) * self.mic_channels
                    mic = mic[:n].reshape(-1, self.mic_channels).mean(axis=1)

                # 2) 回环参考：从队列非阻塞取。
                #    取不到就置零——"回环没有数据"恰好等价于"当前没在播报"，
                #    这是语义正确的降级，而不是错误。
                ref = None
                if self._echo_control_ok:
                    with self._ref_lock:
                        if len(self._ref_queue) >= self.block_samples:
                            ref = self._ref_queue[:self.block_samples]
                            self._ref_queue = self._ref_queue[self.block_samples:]
                        else:
                            ref = np.zeros(self.block_samples, dtype=np.float32)
                            self.stats["ref_padding"] += 1
                    # 回环真的送来声音了，说明播报已经开始：此时才启动标定
                    if self._calib_pending and rms(ref) > 1e-4:
                        self._calib_pending = False
                        self._start_calibration()

                # 4) 标定期：不做消除，只收集数据（此时不判定打断）
                if ref is not None and self._calibrating:
                    self._feed_calibration(mic, ref)

                # 5) 回声消除
                #    关键：参考信号要**延后** delay 个采样再与麦克风配对——
                #    现在听到的回声，是 delay 个采样之前播出去的。
                residual = mic
                if (ref is not None and self.aec is not None and self.aec.calibrated
                        and not self._calibrating):
                    d = int(self.aec.delay_samples)
                    self._ref_delay_line = np.concatenate([self._ref_delay_line, ref])
                    if len(self._ref_delay_line) >= d + self.block_samples:
                        ref_aligned = self._ref_delay_line[:self.block_samples]
                        self._ref_delay_line = self._ref_delay_line[self.block_samples:]
                    else:
                        ref_aligned = np.zeros(self.block_samples, dtype=np.float32)
                    residual = self.aec.process(mic, ref_aligned)
                    self.stats["erle_db"] = self.aec.last_erle_db

                # 6) 电平统计（界面音量条）
                self.stats["level"] = rms(mic)
                self.stats["residual_level"] = rms(residual)
                blocks += 1
                self.stats["blocks"] = blocks

                # 7) 播报中：判定打断；否则：VAD 切句
                if self._playing:
                    if self.barge_in is not None and not self._calibrating:
                        fired, ratio = self.barge_in.update(residual, playing=True)
                        if fired:
                            self.stats["barge_ins"] += 1
                            if self.on_barge_in:
                                self.on_barge_in()
                            # 打断一次后不再重复报，等 set_playing(False)
                            self._playing = False
                else:
                    if self.barge_in is not None:
                        self.barge_in.update(residual, playing=False)
                    for utt in self.segmenter.push(float_to_pcm(residual)):
                        if utt:
                            self.stats["utterances"] += 1
                            if self.on_utterance:
                                self.on_utterance(utt)
        except Exception as e:
            self.error = "采集线程异常: %s" % e
        finally:
            pass

    def _emit_status(self, text):
        if self.on_status:
            try:
                self.on_status(text)
            except Exception:
                pass


# ======================================================================
# 设备自检（不出声、不落盘，只读电平与格式）
# ======================================================================

def probe(seconds=2.0):
    """
    安全探针：打开麦克风与回环，检查能否正常出块、回环采样率与电平。
    不播放任何声音，不保存任何音频。
    """
    import pyaudio
    print("=== 设备探针（不出声、不落盘；只读电平与格式）===", flush=True)

    print("\n【1】回环设备")
    devs = list_loopback_devices()
    if devs and "error" in devs[0]:
        print("   不可用: %s" % devs[0]["error"])
    else:
        for d in devs:
            print("   [%d] %s (%d Hz)" % (d["index"], d["name"], d["rate"]))

    print("\n【2】麦克风")
    pa = pyaudio.PyAudio()
    try:
        try:
            default_in = pa.get_default_input_device_info()
            print("   默认输入: [%d] %s" % (int(default_in["index"]), default_in["name"]))
        except Exception as e:
            print("   取默认输入失败: %s" % e)
        rate = RECORD_RATE
        block = int(rate * VAD_FRAME_MS / 1000.0)
        stream = pa.open(format=8, channels=1, rate=rate, input=True,
                         frames_per_buffer=block)
        levels = []
        t0 = time.time()
        n = 0
        while time.time() - t0 < seconds:
            data = stream.read(block, exception_on_overflow=False)
            levels.append(rms(pcm_to_float(data)))
            n += 1
        stream.stop_stream()
        stream.close()
        elapsed = time.time() - t0
        print("   %d 个块 / %.2f 秒 = 实收 %.1f 块每秒（期望 %.1f）"
              % (n, elapsed, n / elapsed, 1000.0 / VAD_FRAME_MS))
        print("   环境电平 RMS 平均 %.5f，最大 %.5f（安静环境应接近 0）"
              % (float(np.mean(levels)), float(np.max(levels))))
        print("   结论: 麦克风采集链路%s" % ("正常" if n / elapsed > 20 else "偏慢，需排查"))
    except Exception as e:
        print("   麦克风探测失败: %s" % e)
    finally:
        pa.terminate()

    print("\n【3】连续监听链路（不出声，跑 2.5 秒看主循环是否被回环阻塞）")
    lis = ContinuousListener(on_status=lambda t: print("   [状态] %s" % t))
    ok, err = lis.start()
    if not ok:
        print("   启动失败: %s" % err)
    else:
        print("   麦克风: 设备=%s 声道=%s" % (lis.mic_device_used, lis.mic_channels))
        print("   回环: %s" % (lis.loopback_info if lis.loopback_info else "不可用"))
        time.sleep(0.3)                      # 让采集稳定下来
        # 取增量再计时：否则会把启动阶段的块也算进来，速率看起来偏高
        b0 = lis.stats["blocks"]
        r0 = lis.stats.get("ref_blocks", 0)
        t0 = time.time()
        time.sleep(2.5)
        elapsed = time.time() - t0
        blocks = lis.stats["blocks"] - b0
        ref_blocks = lis.stats.get("ref_blocks", 0) - r0
        pad = lis.stats["ref_padding"]
        level = lis.stats["level"]
        lis.stop()
        rate = blocks / elapsed if elapsed > 0 else 0
        print("   %.1f 秒内主循环处理 %d 块 = %.1f 块/秒（期望 %.1f）"
              % (elapsed, blocks, rate, 1000.0 / VAD_FRAME_MS))
        print("   回环回调供块 %d 个；参考信号补零 %d 次" % (ref_blocks, pad))
        print("   麦克风电平 %.5f" % level)
        if rate > 20:
            print("   结论: 主循环正常。回环%s"
                  % ("已供数据" if ref_blocks else
                     "无数据——这是无音频播放时 WASAPI 的正常现象，且已证明它不会阻塞主循环"))
        else:
            print("   结论: 主循环偏慢或被阻塞，需排查")

    print("\n=== 探针结束（未播放声音、未保存音频）===")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        print("本模块是设备层。用法：")
        print("  python -m core.continuous_listener --probe    # 设备探针（不出声、不落盘）")
        print("  from core.continuous_listener import ContinuousListener  # 代码里使用")

# -*- coding: utf-8 -*-
"""
连续语音输入前端：VAD 自动断句 + 回声消除 + 播报中打断

【为什么需要它】这是本项目与"单轮命令式语音查询"的本质区别：
    单轮命令式 —— 用户按一下 → 录一段 → 识别；用户必须盯着按钮。
    连续对话   —— 点一次之后一直听，靠 VAD 自动判断"这句话说完了"。
  但"一直听"会遇到一个死结：**播报时麦克风会听见自己的声音**。
  如果不处理，程序会把自己的播报当成用户插话，逻辑立刻错乱。
  所以回声消除不是加分项，而是连续对话能否成立的前提。

【本模块提供三个可独立测试的部件】
    VadSegmenter    —— 帧级语音活动检测，把连续音频流切成一"句"
    EchoCanceller   —— 块 NLMS 自适应滤波，用回环参考信号抵消扬声器回声
    BargeInDetector —— 在残差信号上判断"用户是否在插话"
    设备接入层（ContinuousListener）待真实麦克风与回环设备调通后再提交，
    因为它的正确性只能靠真机验证，不能靠自检。

【坐标与格式约定】
    统一 16kHz / 16bit / 单声道 PCM（bytes）。webrtcvad 只接受 10/20/30 毫秒帧，
    16kHz 下 30 毫秒 = 480 采样 = 960 字节。
"""

import math
import os
import time
from collections import deque

import numpy as np

try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except Exception:                     # 没装也能用，退化为能量检测
    _HAS_WEBRTCVAD = False

from config import (
    RECORD_RATE,
    VAD_MODE, VAD_FRAME_MS, SILENCE_END_MS, MIN_SPEECH_MS,
    MAX_UTTERANCE_SECONDS, VAD_PADDING_MS,
    VAD_GATE_ENABLED, VAD_GATE_RATIO, VAD_GATE_MIN_RMS,
    AEC_FILTER_MS, AEC_STEP_SIZE,
    BARGE_IN_ENERGY_RATIO, BARGE_IN_MIN_MS, BARGE_IN_GRACE_MS,
)


# ======================================================================
# 工具
# ======================================================================

def pcm_to_float(frame_bytes):
    """16bit PCM bytes -> float32 [-1, 1]"""
    return np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(samples):
    """float32 -> 16bit PCM bytes（带饱和，防溢出爆音）"""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def rms(samples):
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def db_from_ratio(ratio):
    if ratio <= 0:
        return -999.0
    return 10.0 * math.log10(ratio)


def frame_bytes_for(sample_rate, frame_ms):
    return int(sample_rate * frame_ms / 1000.0) * 2   # 16bit = 2 字节


# ======================================================================
# 1) VAD 切句
# ======================================================================

class VadSegmenter:
    """
    把连续音频流切成一"句"。

    状态机：静音 → 检测到语音 → 累积 → 静音超过阈值 → 输出一句
    两个保护：
      - 首尾各留 padding，避免把字头字尾切掉
      - 太短（咳嗽、碰麦）丢弃；太长（用户忘停）强制切分
    """

    def __init__(self, sample_rate=RECORD_RATE, frame_ms=VAD_FRAME_MS, mode=VAD_MODE,
                 silence_end_ms=SILENCE_END_MS, min_speech_ms=MIN_SPEECH_MS,
                 max_utterance_seconds=MAX_UTTERANCE_SECONDS, padding_ms=VAD_PADDING_MS,
                 use_webrtcvad=True):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = frame_bytes_for(sample_rate, frame_ms)
        self.silence_end_frames = max(1, int(silence_end_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.max_frames = max(1, int(max_utterance_seconds * 1000 / frame_ms))
        self.padding_frames = max(1, int(padding_ms / frame_ms))

        self._vad = None
        if use_webrtcvad and _HAS_WEBRTCVAD:
            self._vad = webrtcvad.Vad(int(mode))
        self.backend = "webrtcvad" if self._vad else "energy"

        self._buf = []                 # 当前这句已累积的帧
        self._silence_run = 0          # 连续静音帧数
        self._speech_frames = 0        # 本句里判定为语音的帧数
        self._prefix = deque(maxlen=self.padding_frames)   # 语音起点前的 padding
        self._in_speech = False
        # 长期本底（自适应噪声门用；门默认关闭，见 config.VAD_GATE_ENABLED 的说明）
        self.noise_floor = None
        self.last_dropped_short = 0

    # ---------- 单帧判定 ----------

    def is_speech(self, frame):
        """
        公开的单帧判定，供打断检测复用（判断残差"像不像人声"）。
        纯能量判据会把键盘声、椅子声也当成语音，加上这一道能显著减少误判。
        """
        return bool(self._is_speech(frame))

    def _is_speech(self, frame):
        """
        单帧判定 = webrtcvad 的语音判定 **且** 能量高于自适应噪声门。

        为什么要加能量门：实测本机麦克风本底 RMS 0.0176（-35dBFS 量级），
        webrtcvad 在这么高的本底下会把环境噪声持续判成说话，
        导致（a）切出 6 秒的假句子（b）播报中误触发打断。
        """
        level = rms(pcm_to_float(frame))

        # 长期本底：只在"当前没在说话"时更新，避免说话把本底抬上去。
        # 低于本底时快速跟随（0.9/0.1），高于本底时极慢上升（0.995/0.005）。
        if not self._in_speech:
            if self.noise_floor is None:
                self.noise_floor = level
            elif level < self.noise_floor:
                self.noise_floor = 0.9 * self.noise_floor + 0.1 * level
            else:
                self.noise_floor = 0.995 * self.noise_floor + 0.005 * level

        gate = VAD_GATE_MIN_RMS
        if self.noise_floor is not None:
            gate = max(gate, self.noise_floor * VAD_GATE_RATIO)

        if self._vad is not None:
            if not VAD_GATE_ENABLED or level >= gate:
                try:
                    return self._vad.is_speech(frame, self.sample_rate)
                except Exception:
                    pass
            return False
        return level > gate

    # ---------- 主入口 ----------

    def push(self, frame):
        """
        喂一帧（必须恰好 frame_ms 长）。
        :return: 已完成句子列表（每项是 16bit 单声道 PCM bytes），通常为空或 1 项
        """
        if len(frame) != self.frame_bytes:
            # 容许尾帧不足：补齐到整帧，避免丢尾音
            if len(frame) < self.frame_bytes:
                frame = frame + b"\x00" * (self.frame_bytes - len(frame))
            else:
                frame = frame[:self.frame_bytes]

        done = []
        speech = self._is_speech(frame)

        if not self._in_speech:
            if speech:
                self._in_speech = True
                self._silence_run = 0
                self._speech_frames = 1
                self._buf = list(self._prefix)     # 补上起点前的 padding
                self._buf.append(frame)
                self._prefix.clear()
            else:
                self._prefix.append(frame)
            return done

        # 已在语音中
        self._buf.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        # 结束条件 1：静音够久
        if self._silence_run >= self.silence_end_frames:
            done.append(self._finish())
        # 结束条件 2：单句过长，强制切分（避免用户忘停）
        elif len(self._buf) >= self.max_frames:
            done.append(self._finish())
        return done

    def _finish(self):
        """收尾：裁掉多余尾静音、判断有效长度，返回 PCM 或 b"" """
        buf = self._buf
        trailing_silence = min(self._silence_run, len(buf))
        self._buf = []
        self._in_speech = False
        self._silence_run = 0
        speech_frames = self._speech_frames
        self._speech_frames = 0
        self._prefix.clear()

        if speech_frames < self.min_speech_frames:
            self.last_dropped_short += 1
            return b""
        # 尾部静音只留 padding：既减小上传体积，也避免让 ASR 处理长段死气
        if trailing_silence > self.padding_frames:
            buf = buf[:len(buf) - (trailing_silence - self.padding_frames)]
        return b"".join(buf)

    def flush(self):
        """外部要求立即结束当前句（例如用户点了"停止聆听"）"""
        if self._in_speech and self._buf:
            return self._finish()
        self._buf = []
        self._in_speech = False
        self._prefix.clear()
        return b""

    def reset(self):
        self._buf = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0
        self._prefix.clear()


# ======================================================================
# 2) 回声消除（块 NLMS）
# ======================================================================

class EchoCanceller:
    """
    用"系统正在播放的声音"（WASAPI 回环采集）作为参考信号，
    自适应估计扬声器→麦克风的传输路径，再从麦克风信号里把它减掉。

    实现为块 NLMS：
      y = w * x_ref                      （回声估计，FIR 卷积）
      e = mic - y                        （残差 = 用户真正的声音）
      w += mu * <e, x_ref> / (||x_ref||^2 + eps)   （按块更新）

    为什么先估时延与增益：麦克风与回环是两条独立流，存在固定错位与音量差；
    不先对齐，NLMS 会为了追这个错位而发散。
    """

    def __init__(self, sample_rate=RECORD_RATE, filter_ms=AEC_FILTER_MS,
                 step_size=AEC_STEP_SIZE):
        self.sample_rate = sample_rate
        self.taps = max(16, int(sample_rate * filter_ms / 1000.0))
        self.mu = float(step_size)
        self.w = np.zeros(self.taps, dtype=np.float32)
        # 参考信号缓冲：长度 = taps，最新样本在尾部
        self._xbuf = np.zeros(self.taps, dtype=np.float32)
        self.delay_samples = 0
        self.gain = 1.0
        self.calibrated = False
        self.last_erle_db = 0.0
        self._in_energy = 0.0
        self._out_energy = 0.0

    # ---------- 标定 ----------

    def calibrate(self, mic, ref, max_delay_samples=None):
        """
        用一段"只有回声、没有用户说话"的录音估计时延与增益。
        :param mic: float32 麦克风信号
        :param ref: float32 回环（正在播放）信号
        :return: (delay_samples, gain, peak_corr)
        """
        mic = np.asarray(mic, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        n = min(len(mic), len(ref))
        if n < 1024:
            self.calibrated = False
            return 0, 1.0, 0.0
        mic, ref = mic[:n], ref[:n]
        if max_delay_samples is None:
            max_delay_samples = min(self.taps, n // 2)

        # 用互相关找时延（ref 在前、mic 在后 => mic 落后 ref 若干采样）
        best_delay, best_corr = 0, -1.0
        ref_norm = ref / (np.linalg.norm(ref) + 1e-9)
        for d in range(0, max_delay_samples, 8):        # 粗搜，步长 8
            seg = mic[d:d + len(ref) - max_delay_samples]
            r = ref[:len(seg)]
            if len(seg) < 256:
                break
            c = abs(float(np.dot(r / (np.linalg.norm(r) + 1e-9), seg / (np.linalg.norm(seg) + 1e-9))))
            if c > best_corr:
                best_corr, best_delay = c, d
        # 细搜 ±8
        for d in range(max(0, best_delay - 8), best_delay + 9):
            seg = mic[d:d + len(ref) - max_delay_samples]
            r = ref[:len(seg)]
            if len(seg) < 256:
                break
            c = abs(float(np.dot(r / (np.linalg.norm(r) + 1e-9), seg / (np.linalg.norm(seg) + 1e-9))))
            if c > best_corr:
                best_corr, best_delay = c, d

        self.delay_samples = int(best_delay)
        # 增益用最小二乘：g = <mic, ref_d> / <ref_d, ref_d>
        seg = mic[self.delay_samples:self.delay_samples + len(ref) - max_delay_samples]
        r = ref[:len(seg)]
        denom = float(np.dot(r, r)) + 1e-9
        self.gain = float(np.dot(seg, r) / denom) if len(seg) else 1.0
        self.gain = max(0.0, min(2.0, self.gain))
        self.calibrated = True
        return self.delay_samples, self.gain, best_corr

    # ---------- 逐块处理 ----------

    def _ref_window(self):
        return self._ref_hist

    def process(self, mic_block, ref_block):
        """
        处理一块音频（建议 10~30ms）。
        :param mic_block: float32 麦克风
        :param ref_block: float32 回环参考（与 mic 同步）
        :return: 残差 float32（已减掉回声估计）

        数学约定（这里最容易写错，之前就踩过）：
          参考缓冲 xbuf 长度 M，最新样本在尾部；当前块长度 B。
          回声估计 y[n] = Σ_k w[k] * xbuf[M-B+n-k]，n = 0..B-1
          用 np.convolve(xbuf, w) 的第 M-B .. M-1 项即可对上。
          注意不能用 mode='valid'——xbuf 与 w 等长时它会返回空数组，
          结果 y 恒为 0，表现为"回声一点没消除"。
        """
        mic_block = np.asarray(mic_block, dtype=np.float32)
        ref_block = np.asarray(ref_block, dtype=np.float32)
        n = min(len(mic_block), len(ref_block))
        if n == 0:
            return mic_block
        mic_block, ref_block = mic_block[:n], ref_block[:n]
        M = self.taps
        if n >= M:                      # 块比滤波器还长：只保留尾部 M 个
            mic_block = mic_block[-M:]
            ref_block = ref_block[-M:]
            n = M
        B = n

        # 1) 参考信号入缓冲
        if B >= M:
            self._xbuf = ref_block.copy()
        else:
            self._xbuf = np.concatenate([self._xbuf[B:], ref_block]).astype(np.float32)
        xbuf = self._xbuf

        # 2) 回声估计
        y_full = np.convolve(xbuf, self.w)
        y = y_full[M - B:M]
        e = (mic_block - y).astype(np.float32)

        # 3) 块 NLMS 更新：grad[k] = Σ_n e[n] * xbuf[M-B+n-k]
        #    令 m = M-1-(M-B+n-k) = B-1-n+k，则 xbuf[M-B+n-k] = xr[B-1-n+k]
        #    => 对每个 n，切片是 xr[base : base+L]，base = B-1-n
        #    （切片方向写反会让滤波器往反方向更新，表现为越消越糟、ERLE 变负）
        xr = xbuf[::-1]                 # xr[m] = xbuf[M-1-m]，切片连续、更新更快
        grad = np.zeros(M, dtype=np.float32)
        for i in range(B):
            base = B - 1 - i
            if base < 0:
                continue
            L = M - base                # xr 长度 M，可用下标 base..M-1
            grad[:L] += e[i] * xr[base:base + L]
        norm = float(np.dot(xbuf, xbuf)) + 1e-6
        self.w = (self.w + (self.mu * grad / norm)).astype(np.float32)

        # 4) 数值保护：发散就重置，避免噪声炸开
        if not np.all(np.isfinite(self.w)) or float(np.dot(self.w, self.w)) > 1e4:
            self.w = np.zeros(M, dtype=np.float32)

        # 5) 统计回声抑制量 ERLE
        self._in_energy += float(np.dot(mic_block, mic_block))
        self._out_energy += float(np.dot(e, e))
        if self._in_energy > 1e-9:
            self.last_erle_db = db_from_ratio(self._in_energy / max(self._out_energy, 1e-12))
        return e

    def reset_stats(self):
        self._in_energy = 0.0
        self._out_energy = 0.0
        self.last_erle_db = 0.0

    def reset(self):
        self.w = np.zeros(self.taps, dtype=np.float32)
        self._xbuf = np.zeros(self.taps, dtype=np.float32)
        self.reset_stats()


# ======================================================================
# 3) 播报中打断检测
# ======================================================================

class BargeInDetector:
    """
    判断"用户在播报期间插话"。

    原理：播报时麦克风里主要是回声（已被 EchoCanceller 压掉大部分），
    残差里如果出现持续高于本底的能量，就是用户在说话。

    三个防误触发的措施：
      - grace：起播后一小段保护期不判定（起播瞬态容易误报）
      - 持续时长：必须连续超过 min_ms 才算，咳嗽/碰麦不足以触发
      - 本底自适应：快速下降、缓慢上升，让"稳态回声"的比值落在 1 附近

    【为什么按毫秒计时而不是按帧数】
    调用方每次喂的块长可能不同（10ms/20ms/30ms）。如果按帧数换算，
    一旦块长与假设不符，"持续 300ms 才触发"就会变成 100ms 就触发——
    这正是之前误报的根源。这里一律用采样数换算成毫秒累计。
    """

    def __init__(self, sample_rate=RECORD_RATE, energy_ratio=BARGE_IN_ENERGY_RATIO,
                 min_ms=BARGE_IN_MIN_MS, grace_ms=BARGE_IN_GRACE_MS):
        self.sample_rate = sample_rate
        self.energy_ratio = float(energy_ratio)
        self.min_ms = float(min_ms)
        self.grace_ms = float(grace_ms)
        self._run_ms = 0.0
        self._playing_ms = 0.0
        self._floor = 1e-4          # 残差本底能量
        self.triggered_at = None
        self.last_ratio = 0.0

    def update(self, residual, playing=True, is_speech=None):
        """
        :param residual: 回声消除后的残差（float32）
        :param playing: 当前是否在播报
        :param is_speech: 这一帧是否"像人声"（webrtcvad 判定）。传 False 时
                          即使能量高也不累加——用来挡掉键盘声、椅子声这类噪声。
                          传 None 表示不做语音门控。
        :return: (是否判定打断, 当前能量/本底比值)
        """
        block_ms = (len(residual) / float(self.sample_rate) * 1000.0) if len(residual) else 0.0
        energy = float(np.mean(np.square(residual))) if len(residual) else 0.0

        # 【本底更新的时机是这里最关键的设计】
        # 只在"判定为没有用户语音"的帧更新本底。若在用户说话时也更新，
        # 本底会被语音一起拉高、比值迅速衰减，结果就是"越说话越不触发"。
        if not playing:
            self._floor = 0.95 * self._floor + 0.05 * max(energy, 1e-8)
            self._floor = max(self._floor, 1e-8)
            self._run_ms = 0.0
            self._playing_ms = 0.0
            self.triggered_at = None
            return False, energy / self._floor

        self._playing_ms += block_ms

        # 起播保护期：一边不判定，一边快速学习"当前回声残差有多大"。
        # 没有这一步，残留的大回声会被当成用户插话（第一次版本就栽在这）。
        if self._playing_ms <= self.grace_ms:
            self._floor = 0.7 * self._floor + 0.3 * max(energy, 1e-8)
            self._floor = max(self._floor, 1e-8)
            return False, energy / self._floor

        if energy < self.energy_ratio * self._floor:
            self._floor = 0.95 * self._floor + 0.05 * max(energy, 1e-8)
        self._floor = max(self._floor, 1e-8)
        ratio = energy / self._floor
        self.last_ratio = ratio

        # 带泄漏的累加器：超阈值就累加，低于阈值就扣减而不是清零。
        # 真实语音是音节性的（能量天然起伏），"必须连续超阈值"会被一次低谷清零，
        # 导致该触发时反而永远触发不了（踩过这个坑）。
        # 再加一道语音门控：能量够 + 像人声 才累加。
        voiced = (is_speech is None) or bool(is_speech)
        if ratio >= self.energy_ratio and voiced:
            self._run_ms += block_ms
        else:
            self._run_ms = max(0.0, self._run_ms - block_ms)

        if self._run_ms >= self.min_ms:
            if self.triggered_at is None:
                self.triggered_at = self._playing_ms
            return True, ratio
        return False, ratio

    def reset(self):
        self._run_ms = 0.0
        self._playing_ms = 0.0
        self.triggered_at = None


# ======================================================================
# 4) 回环设备枚举（真机用；只查不改）
# ======================================================================

def list_loopback_devices():
    """
    列出 WASAPI 回环设备（= 能录到"系统正在播放的声音"的设备）。
    需要 pyaudiowpatch。

    不要假设 WASAPI 就是 host API 0——本机就不是（之前因此查出 0 个设备）。
    优先用库自带的生成器，退化方案才是遍历全部设备按 isLoopbackDevice 过滤。
    :return: [{"index":int,"name":str,"rate":int}, ...]
    """
    try:
        import pyaudiowpatch as pyaudio_wp
    except Exception as e:
        return [{"error": "未安装 pyaudiowpatch: %s" % e}]

    out = []
    pa = None
    try:
        pa = pyaudio_wp.PyAudio()

        def _collect(info):
            out.append({
                "index": int(info["index"]),
                "name": str(info.get("name", "")),
                "rate": int(info.get("defaultSampleRate", 0)),
            })

        # 方案一：库提供的回环设备生成器
        try:
            for info in pa.get_loopback_device_info_generator():
                _collect(info)
        except Exception:
            pass

        # 方案二（退化）：遍历所有设备，按标记过滤
        if not out:
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if info.get("isLoopbackDevice"):
                    _collect(info)
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
    return out


# ======================================================================
# 自检：用合成信号验证算法（不需要麦克风与扬声器）
# ======================================================================

def _synth_speechlike(n, seed=0):
    """合成"类语音"信号：带包络的带限噪声，用来模拟播报内容"""
    rng = np.random.RandomState(seed)
    x = rng.randn(n).astype(np.float32)
    # 简单低通，去掉高频毛刺，更像语音
    k = np.ones(9, dtype=np.float32) / 9.0
    x = np.convolve(x, k, mode="same")
    # 音节包络：约 4Hz 的起伏
    t = np.arange(n) / 16000.0
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    return (x * env * 0.5).astype(np.float32)


if __name__ == "__main__":
    print("=== 音频前端自检（合成信号，不占用你的麦克风/扬声器）===")
    print("VAD 后端: %s" % ("webrtcvad" if _HAS_WEBRTCVAD else "energy（未装 webrtcvad）"))

    # ---------- 1) VAD 切句 ----------
    print("\n【1】VAD 自动断句")
    seg = VadSegmenter()
    fb = seg.frame_bytes
    rate = seg.sample_rate
    n_sil = int(0.5 * rate / (fb // 2))      # 0.5 秒静音帧数
    n_sp = int(1.2 * rate / (fb // 2))       # 1.2 秒说话帧数
    speech_sig = _synth_speechlike(n_sp * (fb // 2), seed=1)
    silence = np.zeros(n_sp * (fb // 2), dtype=np.float32)
    frames = ([float_to_pcm(silence[:fb // 2])] * n_sil
              + [float_to_pcm(speech_sig[i * (fb // 2):(i + 1) * (fb // 2)]) for i in range(n_sp)]
              + [float_to_pcm(silence[:fb // 2])] * int(1.2 * rate / (fb // 2)))
    got = []
    for f in frames:
        got.extend([u for u in seg.push(f) if u])
    total_ms = sum(len(u) / 2 / rate * 1000 for u in got)
    print("   输入: 0.5s 静音 + 1.2s 语音 + 1.2s 静音（共 %d 帧）" % len(frames))
    print("   切出句子数: %d，语音总时长 %.0f ms"
          "（期望 ≈ 语音1200 + 首尾padding各200 = 1600）" % (len(got), total_ms))
    ok1 = len(got) == 1 and 1300 <= total_ms <= 1900
    print("   判定: %s" % ("通过" if ok1 else "不通过"))

    # ---------- 2) 回声消除 ----------
    print("\n【2】回声消除（块 NLMS）")
    rate = 16000
    dur = 3.0
    n = int(rate * dur)
    ref = _synth_speechlike(n, seed=7)                      # 正在播放的内容
    delay = int(0.025 * rate)                               # 扬声器→麦克风 25ms
    gain = 0.35                                             # 回声衰减
    echo = np.zeros(n, dtype=np.float32)
    echo[delay:] = ref[:n - delay] * gain
    noise = (np.random.RandomState(3).randn(n) * 0.005).astype(np.float32)
    user = np.zeros(n, dtype=np.float32)
    t0, t1 = int(1.0 * rate), int(1.6 * rate)               # 用户在 1.0~1.6s 插话
    user[t0:t1] = _synth_speechlike(t1 - t0, seed=11) * 0.8
    mic = echo + user + noise

    aec = EchoCanceller()
    cal_len = int(0.8 * rate)                               # 用前 0.8s（只有回声）标定
    d_est, g_est, corr = aec.calibrate(mic[:cal_len], ref[:cal_len])
    print("   真实时延 %d 采样(%.0fms) 增益 %.2f" % (delay, delay / rate * 1000, gain))
    print("   估计时延 %d 采样(%.0fms) 增益 %.2f 相关度 %.3f"
          % (d_est, d_est / rate * 1000, g_est, corr))

    aec.reset()
    # 用估计出的时延对齐后喂给 NLMS
    aligned_mic = mic[d_est:]
    aligned_ref = ref[:len(aligned_mic)]
    block = 160
    residual = np.zeros_like(aligned_mic)
    t_start = time.perf_counter()
    for i in range(0, len(aligned_mic) - block, block):
        residual[i:i + block] = aec.process(aligned_mic[i:i + block],
                                            aligned_ref[i:i + block])
    t_cost = time.perf_counter() - t_start
    audio_sec = len(aligned_mic) / float(rate)
    print("   处理 %.1fs 音频耗时 %.2fs CPU ≈ 单核 %.0f%%（块长 %d 采样，滤波器 %d 抽头）"
          % (audio_sec, t_cost, t_cost / audio_sec * 100, block, aec.taps))
    # 回声抑制量：滤波器需要几百毫秒收敛，因此分开报"含收敛过程"和"收敛后"
    # （真实系统里起播前 300ms 有 grace 保护期，判定打断只看收敛后的稳态）
    a, b = int(0.5 * rate), int(0.9 * rate)
    in_e = float(np.mean(np.square(aligned_mic[a:b])))
    out_e = float(np.mean(np.square(residual[a:b])))
    erle = db_from_ratio(in_e / max(out_e, 1e-12))
    a0, b0 = int(0.2 * rate), int(0.9 * rate)
    in_e0 = float(np.mean(np.square(aligned_mic[a0:b0])))
    out_e0 = float(np.mean(np.square(residual[a0:b0])))
    erle_full = db_from_ratio(in_e0 / max(out_e0, 1e-12))
    print("   回声抑制量 ERLE：收敛后(0.5~0.9s) %.1f dB | 含收敛过程(0.2~0.9s) %.1f dB"
          % (erle, erle_full))
    print("   回声残差能量 %.5f ｜ 插话段残差 %.5f（后者应远大于前者）"
          % (out_e, float(np.mean(np.square(residual[int(1.0 * rate):int(1.6 * rate)])))))
    ok2 = erle >= 10.0
    print("   判定: %s（收敛后 ≥10dB）" % ("通过" if ok2 else "不通过"))

    # ---------- 3) 打断检测 ----------
    print("\n【3】播报中打断检测")
    bid = BargeInDetector()
    trig_ms = None
    false_before = 0
    ratio_echo = []
    ratio_user = []
    for i in range(0, len(residual) - block, block):
        seg_res = residual[i:i + block]
        fired, ratio = bid.update(seg_res, playing=True)
        now_ms = i / rate * 1000
        if 200 <= now_ms <= 900:
            ratio_echo.append(ratio)
        if 1000 <= now_ms <= 1600:
            ratio_user.append(ratio)
        if fired and trig_ms is None:
            trig_ms = now_ms
        if now_ms < 1000 and fired:
            false_before += 1
    print("   回声段(0.2~0.9s) 能量/本底 比值均值 %.2f" % (np.mean(ratio_echo) if ratio_echo else 0))
    print("   插话段(1.0~1.6s) 能量/本底 比值均值 %.2f" % (np.mean(ratio_user) if ratio_user else 0))
    print("   用户真实插话起点: 1000 ms")
    print("   触发时刻: %s" % ("%.0f ms" % trig_ms if trig_ms is not None else "未触发"))
    print("   插话前误触发次数: %d" % false_before)
    ok3 = trig_ms is not None and 900 <= trig_ms <= 1500 and false_before == 0
    print("   判定: %s" % ("通过" if ok3 else "不通过"))

    # ---------- 4) 回环设备 ----------
    print("\n【4】本机 WASAPI 回环设备")
    devs = list_loopback_devices()
    if devs and "error" in devs[0]:
        print("   查询失败: %s" % devs[0]["error"])
    else:
        for d in devs[:5]:
            print("   [%d] %s (%d Hz)" % (d["index"], d["name"], d["rate"]))
        print("   共 %d 个回环设备" % len(devs))

    print("\n=== 汇总: 断句 %s | 回声消除 %s | 打断 %s ==="
          % ("通过" if ok1 else "不通过", "通过" if ok2 else "不通过",
             "通过" if ok3 else "不通过"))

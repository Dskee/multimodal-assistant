# -*- coding: utf-8 -*-
"""
录音模块 Recorder
使用 pyaudio 从麦克风采集音频，输出 16bit 单声道 16kHz WAV 到内存（不依赖 ffmpeg）

设计要点：
  1. 优先按 config 的声道数打开；失败时自动尝试 2 声道 / 1 声道，并自动探测设备
  2. 无论录到几声道，统一降混成"单声道 16bit PCM"，保证 ASR 一定能识别
  3. 录音质量检测：完全静音 / 音量过低 会在本地拦截，避免浪费 API 调用
"""

import io
import wave
import time
import threading

import numpy as np
import pyaudio

from config import (
    RECORD_FORMAT,
    RECORD_CHANNELS,
    RECORD_RATE,
    RECORD_CHUNK,
    RECORD_MAX_SECONDS,
    RECORD_DEVICE,
)


class Recorder:
    """
    录音机类
    用法：
        rec = Recorder()
        rec.start()              # 开始录音
        ...                      # 录音中
        audio_buf, err = rec.stop()  # 停止录音，返回 WAV BytesIO
    """

    def __init__(self):
        self.is_recording = False          # 是否正在录音
        self._audio_data = []              # 存放录音原始帧数据
        self._stream = None                # pyaudio 的输入流
        self._pyaudio = None               # pyaudio 实例
        self._thread = None                # 录音线程
        self._error_msg = ""               # 错误信息
        self._actual_channels = 1          # 实际打开成功的声道数
        self._actual_device = None         # 实际使用的设备索引

    # ==================== 录音控制 ====================

    def start(self):
        """
        开始录音
        返回: (是否成功, 错误信息)
        """
        if self.is_recording:
            return False, "已经在录音中了"

        self._audio_data = []
        self._error_msg = ""

        try:
            self._pyaudio = pyaudio.PyAudio()

            # 依次尝试多种 (声道数, 设备) 组合，只要有一种能打开就用
            self._stream = None
            for channels, device_idx in self._build_open_candidates():
                try:
                    kwargs = {}
                    if device_idx is not None:
                        kwargs['input_device_index'] = device_idx
                    stream = self._pyaudio.open(
                        format=RECORD_FORMAT,
                        channels=channels,
                        rate=RECORD_RATE,
                        input=True,
                        frames_per_buffer=RECORD_CHUNK,
                        **kwargs,
                    )
                    self._stream = stream
                    self._actual_channels = channels
                    self._actual_device = device_idx
                    if device_idx is not None and channels != RECORD_CHANNELS:
                        print(f"⚠️ 声道/设备自动降级: {channels}ch, device={device_idx}")
                    break
                except Exception:
                    continue

            if self._stream is None:
                raise Exception("所有声道/设备组合都打开失败，请检查麦克风是否被占用或未连接")

            self.is_recording = True

            # 后台线程持续采集音频，避免阻塞调用者
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()

            return True, ""

        except Exception as e:
            self._error_msg = f"麦克风打开失败: {str(e)}"
            self.is_recording = False
            self._cleanup()
            return False, self._error_msg

    def _build_open_candidates(self):
        """
        构造尝试列表：(声道数, 设备索引)
        优先级：用户指定设备 > 用户指定声道 > 双声道 > 单声道 > 默认设备
        """
        candidates = []

        if RECORD_DEVICE is not None:
            candidates.append((RECORD_CHANNELS, RECORD_DEVICE))
            other_ch = 1 if RECORD_CHANNELS == 2 else 2
            candidates.append((other_ch, RECORD_DEVICE))

        candidates.append((RECORD_CHANNELS, None))
        for ch in (2, 1):
            if ch != RECORD_CHANNELS:
                candidates.append((ch, None))

        # 去重并保持顺序
        seen = set()
        result = []
        for item in candidates:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _record_loop(self):
        """
        后台录音循环：持续从麦克风读取音频帧，直到 stop() 被调用或超时
        """
        start_time = time.time()

        while self.is_recording:
            # 超时自动停止（防止用户忘记点停止）
            if time.time() - start_time > RECORD_MAX_SECONDS:
                self.is_recording = False
                break

            try:
                data = self._stream.read(RECORD_CHUNK, exception_on_overflow=False)
                self._audio_data.append(data)
            except Exception as e:
                self._error_msg = f"录音中出错: {str(e)}"
                break

    def stop(self):
        """
        停止录音并返回 WAV 字节数据（io.BytesIO 对象）
        返回: (音频字节 BytesIO, 错误信息)
        """
        if not self.is_recording:
            return None, "当前没有在录音"

        # 告诉录音线程停止
        self.is_recording = False

        # 等待录音线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        # 先保留错误信息再清理资源
        error = self._error_msg

        # 关闭 pyaudio 流
        self._cleanup()

        if error:
            return None, error

        if len(self._audio_data) == 0:
            return None, "没有录到任何声音"

        # 把所有帧拼接成完整的音频字节
        raw_audio = b"".join(self._audio_data)

        # ===== 录音质量检测 =====
        # 用"实际声道数"解析样本，判断是否真的有人声
        actual_ch = self._actual_channels
        try:
            samples = np.frombuffer(raw_audio, dtype=np.int16)
            if actual_ch > 1:
                n = (samples.size // actual_ch) * actual_ch
                samples = samples[:n].reshape(-1, actual_ch)[:, 0]

            peak = float(np.max(np.abs(samples))) if samples.size else 0.0

            # 检测 1: 完全静音
            if peak < 5:
                return None, "麦克风没有采集到任何声音，请检查麦克风是否连接/静音"

            # 检测 2: 音量极低（可能没对准麦克风）
            if peak < 500:
                return None, "录音音量太低，请靠近麦克风再试一次"
        except Exception:
            # 检测逻辑不应影响主流程
            pass

        # ===== 统一转成 16bit 单声道 PCM =====
        # 双声道 → 左右声道取平均（降混）；单声道直接用
        try:
            samples = np.frombuffer(raw_audio, dtype=np.int16)
            if actual_ch > 1:
                n = (samples.size // actual_ch) * actual_ch
                samples = samples[:n].reshape(-1, actual_ch).mean(axis=1).astype(np.int16)
            final_pcm = samples.tobytes()

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)        # 统一单声道
                wf.setsampwidth(2)        # 16bit = 2 字节
                wf.setframerate(RECORD_RATE)  # 16000 Hz
                wf.writeframes(final_pcm)
            wav_buffer.seek(0)

            return wav_buffer, ""

        except Exception as e:
            return None, f"音频转换失败: {str(e)}"

    def get_audio_bytes(self):
        """
        对外接口：获取音频字节（等价于 stop）
        """
        return self.stop()

    # ==================== 资源清理 ====================

    def _cleanup(self):
        """释放 pyaudio 资源"""
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass

        try:
            if self._pyaudio:
                self._pyaudio.terminate()
        except Exception:
            pass

        self._stream = None
        self._pyaudio = None


# ========================= 测试入口 =========================
if __name__ == "__main__":
    # 简单测试：录 3 秒音然后保存到本地（仅测试用）
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("正在测试录音模块...")
    print("请对着麦克风说话，3 秒后自动停止...")

    rec = Recorder()
    ok, err = rec.start()
    if not ok:
        print(f"启动失败: {err}")
    else:
        time.sleep(3)
        audio_buf, err = rec.stop()
        if audio_buf:
            with open("test_record.wav", "wb") as f:
                f.write(audio_buf.read())
            print("录音成功！已保存到 test_record.wav")
        else:
            print(f"录音失败: {err}")

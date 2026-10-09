# -*- coding: utf-8 -*-
"""
语音识别模块 SoundToText（ASR）
使用硅基流动 SiliconFlow 的 FunAudioLLM/SenseVoiceSmall 模型
把音频文件（BytesIO）转为文字
"""

# 优先加载项目 libs 目录（内置 typing_extensions，修复 base 环境版本过旧导致 openai 导入失败）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from openai import OpenAI
from config import API_KEY, BASE_URL, STT_MODEL, ASR_TIMEOUT_SECONDS


class SoundToText:
    """
    语音识别类
    用法：
        stt = SoundToText(api_key="你的key")
        text, err = stt.transcribe(audio_bytes_io)
    """

    def __init__(self, api_key=None):
        """
        初始化
        :param api_key: 硅基流动 API Key，不填则用 config.py 中的默认值
        """
        self.api_key = api_key if api_key else API_KEY
        # 同步客户端。
        # 必须带超时：实测 ASR 延迟抖动很大（稳态 0.2~0.4 秒，但出现过 60 秒以上），
        # 不设超时时 SDK 默认 600 秒，一次卡住就把整条连续对话链路挂死。
        self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL,
                             timeout=ASR_TIMEOUT_SECONDS, max_retries=0)
        self.model = STT_MODEL

    def set_api_key(self, key):
        """换 Key 时重建客户端（只改 self.api_key 不会生效）"""
        self.api_key = key or ""
        self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL,
                             timeout=ASR_TIMEOUT_SECONDS, max_retries=0)

    def transcribe_file(self, filename, fileobj, mime):
        """
        通用上传识别：适用于任意音频格式（wav / mp3 / m4a 等）
        :param filename: 上传文件名，如 "audio.wav"、"audio.mp3"
        :param fileobj: 文件对象（io.BytesIO）
        :param mime: MIME 类型，如 "audio/wav"、"audio/mp3"
        :return: (识别文本, 错误信息)
        """
        if not self.api_key:
            return "", "API Key 未填写，请在设置中填入硅基流动 API Key"

        try:
            fileobj.seek(0)
            response = self.client.audio.transcriptions.create(
                file=(filename, fileobj, mime),
                model=self.model,
            )
            return response.text.strip(), ""

        except Exception as e:
            return "", f"语音识别失败: {str(e)}"

    def transcribe(self, audio_buffer):
        """
        调用语音识别接口（Recorder 输出的 16bit 单声道 WAV）
        :param audio_buffer: io.BytesIO 对象（包含 16bit 单声道 WAV 音频数据）
        :return: (识别文本, 错误信息)  识别失败返回 ("", 错误描述)
        """
        return self.transcribe_file("audio.wav", audio_buffer, "audio/wav")


# ========================= 测试入口 =========================
if __name__ == "__main__":
    # 简单测试：读取本地 test_record.wav 文件进行识别
    import io
    import os

    if not os.path.exists("test_record.wav"):
        print("请先运行 recorder.py 生成 test_record.wav")
    else:
        with open("test_record.wav", "rb") as f:
            buf = io.BytesIO(f.read())

        stt = SoundToText()
        text, err = stt.transcribe(buf)
        if err:
            print(f"识别失败: {err}")
        else:
            print(f"识别结果: {text}")

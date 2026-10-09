# -*- coding: utf-8 -*-
"""
语音合成模块 TextToSound（TTS）
支持双提供商：
  1. 本地 edge-tts（免费，效果好）
  2. 云端 SiliconFlow CosyVoice2（需 API Key）

统一异步入口：async run(text) -> bytes
"""

# 优先加载项目 libs 目录（内置 typing_extensions，修复 base 环境版本过旧导致 openai 导入失败）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import asyncio

import edge_tts
from openai import AsyncOpenAI

from config import (
    API_KEY,
    BASE_URL,
    TTS_SILICONFLOW_MODEL,
    TTS_SILICONFLOW_VOICE,
    DEFAULT_LOCAL_VOICE,
    DEFAULT_SPEECH_RATE,
    TTS_LOCAL_SERVICE,
    TTS_SILICONFLOW,
)


class TextToSound:
    """
    语音合成类
    用法（异步）：
        tts = TextToSound(provider=TTS_LOCAL_SERVICE, voice="zh-CN-YunxiNeural")
        audio_bytes = await tts.run("你好")
    """

    def __init__(self, provider=None, voice=None, api_key=None, rate=None):
        """
        初始化
        :param provider: 提供商：TTS_LOCAL_SERVICE 或 TTS_SILICONFLOW
        :param voice: 音色名称
        :param api_key: 云端 API Key（不填用 config 默认）
        :param rate: 语速（edge-tts 语法，如 "-10%"）；适老化默认放慢
        """
        self.provider = provider if provider else TTS_LOCAL_SERVICE
        self.voice = voice if voice else DEFAULT_LOCAL_VOICE
        self.api_key = api_key if api_key else API_KEY
        self.rate = rate if rate else DEFAULT_SPEECH_RATE

    def set_api_key(self, key):
        """换 Key 时同时丢弃已缓存的云端客户端，否则新 Key 不生效"""
        self.api_key = key or ""
        self._silicon_client = None

        # 云端客户端（延迟初始化）
        self._silicon_client = None

    async def run(self, text):
        """
        统一入口：把文字转成 MP3 字节
        最多重试 2 次，失败返回空字节 b""
        :param text: 要合成的文字
        :return: MP3 音频字节（bytes）
        """
        if not text:
            return b""

        for attempt in range(3):  # 尝试 3 次（包括首次）
            try:
                if self.provider == TTS_LOCAL_SERVICE:
                    result = await self._run_edge_tts(text)
                else:
                    result = await self._run_siliconflow(text)

                if result:
                    return result  # 成功，直接返回

            except Exception as e:
                print(f"TTS 第 {attempt + 1} 次尝试失败: {e}")
                await asyncio.sleep(0.5)  # 短暂等待后重试

        # 3 次都失败
        return b""

    # ==================== 本地 edge-tts ====================

    async def _run_edge_tts(self, text):
        """
        使用本地 edge-tts 合成语音
        edge-tts 是异步流式 API，需要收集所有音频块拼接
        """
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch="+0Hz",
            volume="+0%",
        )

        audio_chunks = []

        # 异步迭代接收数据块
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
            # 其他类型比如 "WordBoundary" 可以忽略

        if not audio_chunks:
            raise Exception("edge-tts 没有返回任何音频数据")

        return b"".join(audio_chunks)

    # ==================== 云端 SiliconFlow CosyVoice2 ====================

    async def _run_siliconflow(self, text):
        """
        使用硅基流动 CosyVoice2 合成语音
        """
        if not self.api_key:
            raise Exception("SiliconFlow TTS 需要 API Key")

        # 延迟初始化 AsyncOpenAI 客户端
        if self._silicon_client is None:
            self._silicon_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=BASE_URL,
                timeout=60.0,      # 单次请求 60 秒超时，避免网络卡死
                max_retries=1,     # 最多重试 1 次
            )

        # 调用语音合成接口
        response = await self._silicon_client.audio.speech.create(
            model=TTS_SILICONFLOW_MODEL,
            voice=TTS_SILICONFLOW_VOICE,
            input=text,
            response_format="mp3",
            speed=1.0,
        )

        # 读取完整音频字节
        audio_bytes = await response.aread()

        if not audio_bytes:
            raise Exception("CosyVoice2 返回空音频")

        return audio_bytes


# ========================= 测试入口 =========================
if __name__ == "__main__":
    import tempfile
    import os

    async def test():
        tts = TextToSound(provider=TTS_LOCAL_SERVICE)
        audio = await tts.run("你好，这是一个语音合成测试")
        if audio:
            path = os.path.join("..", "temp", "test_tts.mp3")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(audio)
            print(f"本地 TTS 成功！已保存到 {path}")
        else:
            print("本地 TTS 失败")

    asyncio.run(test())

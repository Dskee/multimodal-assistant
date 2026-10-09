# -*- coding: utf-8 -*-
"""
视觉多模态模块 ImageAnalyzer
使用硅基流动 SiliconFlow 的 Pro/Qwen/Qwen2.5-VL-7B-Instruct 模型
可以从摄像头拍照或读取本地图片，然后用多模态模型分析

支持流式回复（一个 token 一个 token 接收）
"""

# 优先加载项目 libs 目录（内置 typing_extensions，修复 base 环境版本过旧导致 openai 导入失败）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import asyncio
import base64

import cv2
from openai import AsyncOpenAI, APIStatusError

from config import API_KEY, BASE_URL, VISION_MODEL


class ImageAnalyzer:
    """
    图像分析类
    用法（异步）：
        analyzer = ImageAnalyzer()
        # 从摄像头拍照
        frame, err = analyzer.capture_from_camera()
        # 分析图片
        reply, err = await analyzer.analyze_image(frame, "请描述一下这张图片")
    """

    def __init__(self, api_key=None):
        self.api_key = api_key if api_key else API_KEY
        # 延迟初始化 AsyncOpenAI 客户端
        self._client = None
        self.model = VISION_MODEL

    def _get_client(self):
        """延迟创建 AsyncOpenAI 客户端"""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=BASE_URL)
        return self._client

    # ==================== 图片采集 ====================

    def capture_from_camera(self, camera_index=0):
        """
        从摄像头采集一帧
        :param camera_index: 摄像头索引，0 通常是默认摄像头
        :return: (numpy 图片, 错误信息)
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return None, "无法打开摄像头，请检查设备是否连接"

            # 等一会儿让摄像头初始化
            for _ in range(5):
                cap.read()

            ret, frame = cap.read()
            cap.release()  # 立即释放，不要一直占用摄像头

            if not ret or frame is None:
                return None, "摄像头采集失败，请确认权限"

            return frame, ""

        except Exception as e:
            return None, f"摄像头采集异常: {str(e)}"

    def load_from_file(self, file_path):
        """
        从本地文件读取图片
        :param file_path: 图片路径
        :return: (numpy 图片, 错误信息)
        """
        try:
            image = cv2.imread(file_path)
            if image is None:
                return None, f"无法读取图片文件: {file_path}"
            return image, ""
        except Exception as e:
            return None, f"图片读取异常: {str(e)}"

    # ==================== 图片分析 ====================

    async def analyze_image(self, image_np, question="请描述下这张图片", on_chunk=None):
        """
        分析图片（多模态大模型）
        :param image_np: numpy 格式图片（cv2 读取的）
        :param question: 用户对图片的问题
        :param on_chunk: 流式回调函数（每个 token 触发一次）
        :return: (模型回复文本, 错误信息)
        """
        if not self.api_key:
            return "", "API Key 未填写"

        try:
            # 1. 把 numpy 图片编码为 JPEG 字节
            ok, buffer = cv2.imencode('.jpg', image_np)
            if not ok:
                return "", "图片编码失败"

            # 2. 把 JPEG 字节转成 Base64 字符串
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # 3. 构造 data URI
            data_uri = f"data:image/jpeg;base64,{image_base64}"

            # 4. 构造多模态 messages：图片 + 文字
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": question},
                    ]
                }
            ]

            # 5. 调用多模态模型（流式）
            client = self._get_client()
            full_reply = ""

            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=500,           # 视觉模型输出可能较长
                temperature=0.7,
            )

            # 接收流式输出
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_reply += token
                    if on_chunk:
                        on_chunk(token)

            return full_reply.strip(), ""

        except APIStatusError as e:
            # API 特定错误，400 通常是图片格式问题
            if hasattr(e, 'status_code') and e.status_code == 400:
                return "", "传来的图片无法识别，也可能是摄像头没有启动"
            return "", f"视觉模型 API 错误: {str(e)}"

        except Exception as e:
            return "", f"图片分析失败: {str(e)}"


# ========================= 测试入口 =========================
if __name__ == "__main__":
    import sys

    async def test():
        analyzer = ImageAnalyzer()

        # 优先用命令行参数指定的图片，否则用摄像头
        if len(sys.argv) > 1:
            path = sys.argv[1]
            print(f"从文件读取: {path}")
            image, err = analyzer.load_from_file(path)
        else:
            print("尝试从摄像头拍照...")
            image, err = analyzer.capture_from_camera()

        if err:
            print(f"图片获取失败: {err}")
            return

        print("正在分析图片...")
        reply, err = await analyzer.analyze_image(image, on_chunk=lambda t: print(t, end="", flush=True))
        print()
        if err:
            print(f"分析失败: {err}")
        else:
            print(f"\n完整回复: {reply}")

    asyncio.run(test())

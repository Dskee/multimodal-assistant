# -*- coding: utf-8 -*-
"""
大模型对话模块
使用硅基流动 SiliconFlow 的 Qwen2.5-7B-Instruct 模型
支持流式接收回复（一个 token 一个 token 地返回）
"""

# 优先加载项目 libs 目录（内置 typing_extensions，修复 base 环境版本过旧导致 openai 导入失败）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from openai import OpenAI
from config import API_KEY, BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_SYSTEM_PROMPT


class LLM:
    """
    大语言模型对话类
    用法：
        llm = LLM(api_key="你的key")
        # 非流式（一次性拿到完整回复）
        reply, err = llm.chat("你好")
        # 流式（需要回调函数）
        full_reply = llm.chat_stream("你好", on_chunk=lambda chunk: print(chunk, end=""))
    """

    def __init__(self, api_key=None):
        """
        初始化
        :param api_key: 硅基流动 API Key，不填则用 config.py 默认值
        """
        self.api_key = api_key if api_key else API_KEY
        # 同步 OpenAI 客户端
        self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)
        self.model = LLM_MODEL
        # 对话历史（用于多轮对话，简单实现）
        self._history = []

    def _build_messages(self, user_text):
        """
        构造 messages 参数：系统提示词 + 历史对话 + 当前用户输入
        """
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def chat(self, user_text):
        """
        非流式对话（一次性返回完整回复）
        :param user_text: 用户输入
        :return: (AI 回复文本, 错误信息)
        """
        if not self.api_key:
            return "", "API Key 未填写"

        try:
            messages = self._build_messages(user_text)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=False,
            )

            reply_text = response.choices[0].message.content.strip()

            # 更新对话历史
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": reply_text})

            return reply_text, ""

        except Exception as e:
            return "", f"大模型调用失败: {str(e)}"

    def chat_stream(self, user_text, on_chunk=None, context=None, concise_hint=None):
        """
        流式对话（一个 token 一个 token 接收，实时显示）
        :param user_text: 用户输入
        :param on_chunk: 回调函数，每收到一个 token 就调用一次，参数是 token 字符串
        :param context: 联网搜索结果等参考资料（可选），会拼进本轮用户消息
        :param concise_hint: 简洁回答提示（可选），如 100 表示要求回答不超过 100 字
        :return: (完整回复文本, 错误信息)
        """
        if not self.api_key:
            return "", "API Key 未填写"

        try:
            messages = self._build_messages(user_text)
            # 若有联网搜索资料，拼进本轮用户消息（不污染对话历史）
            if context:
                messages[-1]["content"] = (
                    "【联网搜索结果】请优先依据以下资料回答用户问题，"
                    "资料不足或与问题无关时如实说明，不要编造。\n"
                    + context
                    + "\n\n【用户问题】\n"
                    + messages[-1]["content"]
                )
            # 语音场景要求简洁回答（缩短播报时长）
            if concise_hint:
                messages[-1]["content"] += (
                    f"\n（请用不超过 {concise_hint} 字的简洁口语回答，不要列点）"
                )
            full_reply = ""

            # 开启流式模式
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
            )

            # 遍历每个流式 chunk
            for chunk in stream:
                # 确保 delta.content 不为 None
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_reply += token

                    # 如果提供了回调函数，就回调出去
                    if on_chunk:
                        on_chunk(token)

            # 更新对话历史
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": full_reply.strip()})

            return full_reply.strip(), ""

        except Exception as e:
            return "", f"大模型流式调用失败: {str(e)}"

    def clear_history(self):
        """清空对话历史，开始新对话"""
        self._history = []


# ========================= 测试入口 =========================
if __name__ == "__main__":
    print("测试 LLM 流式对话...")

    llm = LLM()
    if not llm.api_key:
        print("请先在 config.py 或代码中填入 API Key")
    else:
        print("AI: ", end="", flush=True)
        reply, err = llm.chat_stream("你好，介绍一下你自己", on_chunk=lambda c: print(c, end="", flush=True))
        print()  # 换行
        if err:
            print(f"错误: {err}")
        else:
            print(f"完整回复: {reply}")

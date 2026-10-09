# -*- coding: utf-8 -*-
"""
文本清理工具

模型天生爱输出 markdown：**加粗**、* 列表、# 标题、`代码`、表情符号。
界面上看起来还行，但**语音播报**会把它念出来——"星号星号"、"井号"、
"反引号"，听起来很糟。所以送进 TTS 之前要统一清一遍。

另外提供"按句子边界截断"：长回答只播报前段，但不能从句子中间切断。
"""

import re

# 常见表情符号范围（够用即可，不求穷尽）
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def strip_markdown(text):
    """
    把 markdown 痕迹清成适合朗读的纯口语文本。
    :param text: 模型原始输出
    :return: 清理后的文本（可能为空串）
    """
    if not text:
        return ""
    s = str(text)

    # 1) 代码块整体去掉（代码没法朗读）
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    # 2) 行内代码去掉反引号，保留内容
    s = s.replace("`", "")
    # 3) 图片/链接：[文字](url) -> 文字
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    # 4) 加粗/斜体标记
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", s)
    # 5) 行首的标题井号与列表符号（含 1. 2) - * • 等）
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s{0,3}[-*+•]\s+", "", s)
    s = re.sub(r"(?m)^\s{0,3}\d{1,2}[.、)）]\s+", "", s)
    # 6) 引用符号
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)
    # 7) 表格分隔线 / 水平线
    s = re.sub(r"(?m)^\s*\|?[\s:|-]{5,}\|?\s*$", " ", s)
    s = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", " ", s)
    # 8) 表格竖线换成逗号停顿
    s = s.replace("|", "，")
    # 9) 表情符号
    s = _EMOJI_RE.sub("", s)
    # 10) 折叠空白：换行变句号停顿，连续空格合一
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "。", s)
    s = s.replace("\n", "，")
    s = re.sub(r"[，。]{2,}", "。", s)
    s = re.sub(r"\s+([，。！？；：])", r"\1", s)
    # 叠在一起的句读（可能隔着空格）收敛成一个句号，避免朗读出多余停顿
    s = re.sub(r"([，。！？；：])(?:\s*[，。！？；：])+", "。", s)
    return s.strip(" ，。、")


def truncate_at_sentence(text, max_chars):
    """
    在句子边界截断，避免播报中途断句。
    找不到合适边界时退回硬截断。
    """
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    idx = max(cut.rfind("。"), cut.rfind("！"), cut.rfind("？"),
              cut.rfind("；"), cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if idx >= max_chars * 0.5:
        return cut[:idx + 1]
    return cut


def prepare_for_speech(text, max_chars=None):
    """一步到位：清理 markdown + 按句子边界截断"""
    cleaned = strip_markdown(text)
    if max_chars:
        cleaned = truncate_at_sentence(cleaned, max_chars)
    return cleaned


# ========================= 测试入口 =========================
if __name__ == "__main__":
    raw = """**今日复习计划** 📅

1. **上午**：过一遍 `极限` 与导数公式
2. 下午：做真题 —— [错题本](http://example.com) 里的题

> 记得早点睡

```python
print("代码不该被念出来")
```

| 科目 | 时间 |
|------|------|
| 高数 | 2h |
"""
    print("=== 原始 ===")
    print(raw)
    print("=== 清理后（准备朗读） ===")
    out = prepare_for_speech(raw, 60)
    print(out)
    print("=== 校验 ===")
    for bad in ("*", "#", "`", "|", "📅"):
        print("  含 %r : %s" % (bad, bad in out))
    print("  长度:", len(out), "（上限 60）")

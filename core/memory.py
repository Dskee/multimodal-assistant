# -*- coding: utf-8 -*-
"""
记忆能力 / 日志助手

【做什么】
  把用户说过的话记下来，并在之后的对话里用得上：
    1. 日志：用户说「记一下 今天血压 130/85」→ 带时间存下来，可回看、可检索；
    2. 关于用户的事实：「记住 我女儿叫小芳」→ 存为长期事实，
       之后问「我女儿叫什么」就能答上来（事实会被注入大模型的上下文）；
    3. 对话日志：每轮对话自动留档，于是「我今天聊了什么」也答得出来。

【为什么这么实现（简历上要能讲清楚）】
  · **纯本地 JSON 文件**，不依赖向量数据库、不联网：数据留在用户自己电脑上，
    这对老年用户的隐私顾虑很重要，也省掉一个重依赖；
  · **检索用字符二元组重合度**（2-gram overlap）：中文不需要分词器，
    十几个字的短句检索效果够用，而且打分过程可解释、可调试。
    数据量上来（上万条）再换 embedding 检索才有意义，现在换是过度设计；
  · 记忆内容**注入到大模型的用户消息里**（不改对话历史），
    这样既能个性化回答，又不会让历史越长越乱。

【存储】
  data/memory.json —— 条目格式：
    {"ts": "2026-01-01 10:00:00", "kind": "log|fact|chat", "text": "...", "reply": "..."}
"""

import json
import os
import re
import time

from config import MEMORY_PATH, MEMORY_ENABLED, MEMORY_MAX_ENTRIES, MEMORY_CONTEXT_ITEMS

KIND_LOG = "log"      # 用户明确要求记下的日志/待办
KIND_FACT = "fact"    # 关于用户的长期事实
KIND_CHAT = "chat"    # 对话自动留档

# 口语里的高频词，参与检索会严重污染结果：
# 例如问「今天天气怎么样」时，"今天"撞上记录「今天血压 130/85」，
# 1/6 的重合度刚好越过阈值，于是把无关的血压记录也注入了（自检时抓到）。
# 去掉这些词再算重合度，上面那次注入就变成 0 命中，而「我上次血压多少」
# 仍能靠"血压"命中——**抓的是内容词，不是寒暄词**。
STOP_BIGRAMS = {
    "今天", "明天", "昨天", "早上", "下午", "晚上", "现在", "什么", "怎么",
    "可以", "一下", "一个", "我们", "你们", "他们", "这个", "那个", "然后",
    "就是", "时候", "多少", "知道", "帮我", "请帮", "记得", "一下", "有没有",
}


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _bigrams(text):
    """字符二元组集合：中文短句检索不需要分词器，二元组重合度就够用"""
    t = re.sub(r"[\s，。！？、,.!?；;：:]+", "", text or "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _score(query, text):
    """
    查询与条目的相关度（0~1）：**去掉口语常见词之后**二元组交集占查询的比例。
    至少要有 1 个非常见词重合才算命中，否则返回 0。
    """
    q = _bigrams(query) - STOP_BIGRAMS
    if not q:
        return 0.0
    t = _bigrams(text) - STOP_BIGRAMS
    inter = q & t
    if not inter:
        return 0.0
    return len(inter) / float(len(q))


class MemoryStore:
    """
    使用：
        mem = MemoryStore()
        mem.add("今天血压 130/85")          # 日志
        mem.add_fact("我女儿叫小芳")         # 长期事实
        mem.add_chat("我头晕", "先坐下休息")  # 对话留档
        mem.recent(5)                       # 最近几条日志
        mem.by_date("today")                # 某天的记录
        mem.search("血压")                  # 检索
        mem.context_for("我上次血压多少")     # 给大模型的上下文
    """

    def __init__(self, path=MEMORY_PATH, enabled=MEMORY_ENABLED):
        self.path = path
        self.enabled = bool(enabled)
        self.entries = []
        self._dirty = False
        self.load()

    # -------------------- 读写 --------------------

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = data.get("entries", [])
                self.entries = [e for e in data if isinstance(e, dict) and e.get("text")]
        except Exception:
            self.entries = []
        self._migrate_old_notes()
        return self.entries

    def _migrate_old_notes(self):
        """
        把旧版「待办」文件（data/notes.json）里的条目并进来。
        旧格式：[{"ts":..., "text":...}]。只做一次：迁移后改名保留，不删用户数据。
        """
        old = os.path.join(os.path.dirname(self.path), "notes.json")
        if not os.path.exists(old):
            return
        try:
            with open(old, "r", encoding="utf-8") as f:
                items = json.load(f)
            added = 0
            for it in items if isinstance(items, list) else []:
                text = (it or {}).get("text")
                if text:
                    self.entries.append({"ts": it.get("ts") or _now(),
                                         "kind": KIND_LOG, "text": text})
                    added += 1
            if added:
                os.rename(old, old + ".migrated")
                self._dirty = True
        except Exception:
            pass

    def save(self):
        if not self._dirty:
            return True
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # 只保留最近 MEMORY_MAX_ENTRIES 条，避免文件无限增长
            if len(self.entries) > MEMORY_MAX_ENTRIES:
                self.entries = self.entries[-MEMORY_MAX_ENTRIES:]
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)      # 原子替换：中途崩溃也不会写坏原文件
            self._dirty = False
            return True
        except Exception:
            return False

    # -------------------- 写入 --------------------

    def _append(self, kind, text, reply=None):
        text = (text or "").strip()
        if not text or not self.enabled:
            return None
        entry = {"ts": _now(), "kind": kind, "text": text}
        if reply:
            entry["reply"] = reply
        self.entries.append(entry)
        self._dirty = True
        self.save()
        return entry

    def add(self, text):
        """日志/待办"""
        return self._append(KIND_LOG, text)

    def add_fact(self, text):
        """关于用户的长期事实"""
        return self._append(KIND_FACT, text)

    def add_chat(self, user_text, reply):
        """对话自动留档"""
        return self._append(KIND_CHAT, user_text, reply=reply)

    # -------------------- 查询 --------------------

    def count(self, kind=None):
        return len(self._filter(kind))

    def _filter(self, kind=None):
        if kind is None:
            return self.entries
        return [e for e in self.entries if e.get("kind") == kind]

    def recent(self, n=5, kind=KIND_LOG):
        items = self._filter(kind)
        return items[-max(1, int(n)):]

    def by_date(self, day, kind=None):
        """
        :param day: "today" / "yesterday" / "2026-01-01" / "week"（最近7天）
        """
        items = self._filter(kind)
        now = time.time()
        if day in ("today", "今天"):
            prefix = time.strftime("%Y-%m-%d")
            return [e for e in items if e.get("ts", "").startswith(prefix)]
        if day in ("yesterday", "昨天"):
            prefix = time.strftime("%Y-%m-%d", time.localtime(now - 86400))
            return [e for e in items if e.get("ts", "").startswith(prefix)]
        if day in ("week", "上周", "这周", "最近一周", "七天"):
            return [e for e in items
                    if self._age_days(e) <= 7]
        # 显式日期（支持 2026-01-01 或 1月1日 这种说法由上层转换）
        return [e for e in items if e.get("ts", "").startswith(day)]

    @staticmethod
    def _age_days(entry):
        try:
            t = time.mktime(time.strptime(entry.get("ts", ""), "%Y-%m-%d %H:%M:%S"))
        except Exception:
            return 9999
        return (time.time() - t) / 86400.0

    def search(self, keyword, limit=5, kind=None):
        """按相关度检索（二元组重合度），返回 [(得分, 条目)]"""
        scored = []
        for e in self._filter(kind):
            s = _score(keyword, e.get("text", ""))
            if s > 0.15:
                scored.append((s, e))
        scored.sort(key=lambda x: (-x[0], x[1].get("ts", "")))
        return scored[:max(1, int(limit))]

    def facts(self, limit=8):
        return self._filter(KIND_FACT)[-max(1, int(limit)):]

    # -------------------- 给大模型的上下文 --------------------

    def context_for(self, text):
        """
        组装注入大模型的记忆上下文：
          长期事实（全部，最多 8 条）+ 与当前问题相关的记录（最多若干条）
        无关时返回空串，避免每次对话都塞一堆没用的信息。
        """
        if not self.enabled or not self.entries:
            return ""
        parts = []
        facts = self.facts()
        if facts:
            parts.append("关于用户已知的事实：" +
                         "；".join(e["text"] for e in facts))
        hits = self.search(text, limit=MEMORY_CONTEXT_ITEMS, kind=KIND_LOG)
        chats = self.search(text, limit=2, kind=KIND_CHAT)
        lines = []
        for s, e in (hits + chats):
            line = "[%s] %s" % (e.get("ts", "")[:16], e.get("text", "")[:60])
            if e.get("reply"):
                line += "（当时回答：%s）" % e["reply"][:40]
            if line not in lines:
                lines.append(line)
        if lines:
            parts.append("与当前问题可能相关的历史记录：\n" + "\n".join(lines[:MEMORY_CONTEXT_ITEMS + 2]))
        return "\n".join(parts)

    # -------------------- 维护 --------------------

    def clear(self, kind=None):
        if kind is None:
            n = len(self.entries)
            self.entries = []
        else:
            n = len(self._filter(kind))
            self.entries = [e for e in self.entries if e.get("kind") != kind]
        self._dirty = True
        self.save()
        return n

    def stats(self):
        return {
            "total": len(self.entries),
            "log": self.count(KIND_LOG),
            "fact": self.count(KIND_FACT),
            "chat": self.count(KIND_CHAT),
            "path": self.path,
        }


if __name__ == "__main__":
    import tempfile
    import shutil

    print("=== 记忆 / 日志助手 自检（用临时文件，不碰真实数据）===")
    tmpdir = tempfile.mkdtemp(prefix="mem_selftest_")
    mem = MemoryStore(path=os.path.join(tmpdir, "memory.json"))

    print("\n【1】写入")
    mem.add("今天血压 130/85，早上吃了降压药")
    mem.add("下午去公园走了四十分钟")
    mem.add("明天上午九点去医院拿体检报告")
    mem.add_fact("我女儿叫小芳，电话 13800000000")
    mem.add_fact("我对青霉素过敏")
    mem.add_chat("我头晕", "先坐下休息，别急着站起来")
    for k, v in mem.stats().items():
        print("  %-6s %s" % (k, v))

    print("\n【2】最近的日志（默认取日志类）")
    for e in mem.recent(3):
        print("  [%s] %s" % (e["ts"][:16], e["text"]))

    print("\n【3】检索（二元组重合度）")
    for q in ["血压", "医院", "女儿", "公园走了多久", "青霉素"]:
        hits = mem.search(q, limit=2)
        if hits:
            print("  查「%s」 -> %.2f  %s" % (q, hits[0][0], hits[0][1]["text"][:38]))
        else:
            print("  查「%s」 -> 无命中" % q)

    print("\n【4】按日期")
    print("  今天 %d 条 / 昨天 %d 条 / 最近七天 %d 条"
          % (len(mem.by_date("today")), len(mem.by_date("yesterday")),
             len(mem.by_date("week"))))

    print("\n【5】注入大模型的上下文（这就是「它有记忆」的关键）")
    print("    设计：长期事实**始终**注入（随时可用）；历史记录**仅相关时**注入")
    for q, expect in [("我女儿叫什么", "相关记录:无（靠事实答）"),
                      ("我上次血压多少", "相关记录:有（血压那条）"),
                      ("今天天气怎么样", "相关记录:无（不干扰）"),
                      ("我对什么过敏", "相关记录:无（靠事实答）")]:
        ctx = mem.context_for(q)
        has_fact = "关于用户已知的事实" in ctx
        has_rec = "与当前问题可能相关" in ctx
        flag = "OK " if ("相关记录:有" in expect) == has_rec else "×  "
        print("  %s 问题：%-14s 事实:%s 相关记录:%s   （期望 %s）"
              % (flag, q, "有" if has_fact else "无", "有" if has_rec else "无", expect))
        if has_rec:
            rec = ctx.split("与当前问题可能相关")[1].replace("\n", " ")[:90]
            print("        命中：%s" % rec)

    print("\n【6】持久化：重新加载后数据还在吗")
    mem2 = MemoryStore(path=mem.path)
    print("  重新加载后条目数：%d（写入时 %d）" % (len(mem2.entries), len(mem.entries)))
    print("  事实条数：%d" % mem2.count("fact"))

    print("\n【7】清空")
    print("  清空日志类 %d 条，剩余 %d 条（事实应保留）" % (mem2.clear("log"), len(mem2.entries)))

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\n自检完成（临时数据已清理）")

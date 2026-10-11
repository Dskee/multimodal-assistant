# -*- coding: utf-8 -*-
"""
本地指令层：免手操作电脑（打开程序、调音量、记事、读剪贴板、读屏幕文字、报时间）

【为什么单独一层、且不经过大模型】
  1. 这类操作必须**立刻**执行：走一次大模型要多等 1~3 秒，而"声音大一点"这种
     指令延迟一秒都很别扭；
  2. 本地规则**可预测、可解释**。绝不能把模型输出直接当成命令去执行——
     模型一旦幻觉出"打开 C:\\某敏感程序"，后果不可控；
  3. 打开程序属于有风险的动作，必须走**白名单**：只允许表里出现过的名字。

【分工】
  命中 → 本地直接执行并返回要播报的话；
  未命中 → 返回 None，交给大模型正常对话。
  因此"帮我看看这个药一天吃几次"仍然走大模型，而"声音大一点"走本地。

【平台】面向 Windows：调音量用 WScript.Shell 发音量键（无需额外依赖）；
读剪贴板用 PowerShell 的 Get-Clipboard；截屏用 Pillow（Anaconda 自带）。
"""

import json
import os
import re
import subprocess
import time

from config import (
    PROJECT_ROOT, COMMANDS_ENABLED, COMMAND_APP_WHITELIST, COMMAND_SITE_WHITELIST,
    COMMAND_VOLUME_KEY_PRESSES, NOTES_PATH,
)

# 音量键：Windows 的虚拟键码
_VK_VOLUME_MUTE = 173
_VK_VOLUME_DOWN = 174
_VK_VOLUME_UP = 175


class CommandResult:
    __slots__ = ("handled", "reply", "action")

    def __init__(self, handled=False, reply="", action=""):
        self.handled = handled
        self.reply = reply
        self.action = action


def _powershell(script, timeout=8):
    """跑一段 PowerShell，返回 (stdout, error)"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0:
            return "", (proc.stderr or "").strip()[:120] or "PowerShell 返回非零"
        return (proc.stdout or "").strip(), ""
    except Exception as e:
        return "", str(e)[:120]


class LocalCommands:
    """
    本地指令匹配与执行。

    用法：
        cmds = LocalCommands(analyzer=..., api_key=...)
        res = cmds.try_handle("声音大一点")
        if res.handled:
            speak(res.reply)
        else:
            交给大模型
    """

    def __init__(self, analyzer=None, api_key=None, notes_path=NOTES_PATH, dry_run=False):
        self.analyzer = analyzer
        self.api_key = api_key
        self.notes_path = notes_path
        self.dry_run = dry_run          # 自检用：只判断"会执行什么"，不真的执行
        self.last_plan = ""             # 自检用：记录将要执行的动作
        self.enabled = bool(COMMANDS_ENABLED)

    # -------------------- 入口 --------------------

    def try_handle(self, text):
        """
        :return: CommandResult；handled=False 表示不归本地管，交给大模型
        """
        if not self.enabled:
            return CommandResult()
        t = (text or "").strip()
        if not t:
            return CommandResult()
        plain = self._plain(t)

        for handler in (self._h_help, self._h_time, self._h_volume, self._h_open,
                        self._h_note, self._h_read_notes, self._h_clipboard,
                        self._h_screen_text):
            res = handler(t, plain)
            if res is not None and res.handled:
                return res
        return CommandResult()

    @staticmethod
    def _plain(text):
        """去掉标点与空格，便于匹配（"声音，大一点！" -> "声音大一点"）"""
        return re.sub(r"[\s，。！？、,.!?~～]", "", text or "")

    # -------------------- 各指令 --------------------

    def _h_help(self, t, p):
        if any(k in p for k in ("你能做什么", "你会什么", "有什么功能", "帮助", "怎么用", "能干啥")):
            apps = "、".join(COMMAND_APP_WHITELIST.keys())
            return CommandResult(True,
                                 "我可以陪您说话，也能帮您做一些操作。"
                                 "您可以说：声音大一点、小一点、静音；"
                                 "打开%s；几点了；记一下…；读一下待办；"
                                 "读一下剪贴板；读一下屏幕上的字。" % apps,
                                 "help")
        return None

    def _h_time(self, t, p):
        if any(k in p for k in ("几点", "时间", "星期几", "几号", "日期", "今天星期")):
            now = time.localtime()
            week = "一二三四五六日"[now.tm_wday]
            if "星期" in p or "周几" in p:
                return CommandResult(True, "今天是星期%s。" % week, "time")
            if any(k in p for k in ("几号", "日期", "今天")):
                return CommandResult(True, "今天是%d月%d日，星期%s。"
                                     % (now.tm_mon, now.tm_mday, week), "time")
            return CommandResult(True, "现在是%d点%d分。"
                                 % (now.tm_hour, now.tm_min if now.tm_min else 0), "time")
        return None

    def _h_volume(self, t, p):
        if "静音" in p:
            mute = not any(k in p for k in ("取消静音", "别静音", "恢复声音", "取消"))
            self.last_plan = "mute" if mute else "unmute"
            if not self.dry_run:
                self._volume_key(_VK_VOLUME_MUTE, 1)
            return CommandResult(True, "好，已静音。" if mute else "好，已取消静音。",
                                 "volume_mute" if mute else "volume_unmute")
        if any(k in p for k in ("声音大", "音量大", "大一点", "大声", "调大")):
            self.last_plan = "volume_up"
            if not self.dry_run:
                self._volume_key(_VK_VOLUME_UP, COMMAND_VOLUME_KEY_PRESSES)
            return CommandResult(True, "好，声音调大了。", "volume_up")
        if any(k in p for k in ("声音小", "音量小", "小一点", "小声", "调小")):
            self.last_plan = "volume_down"
            if not self.dry_run:
                self._volume_key(_VK_VOLUME_DOWN, COMMAND_VOLUME_KEY_PRESSES)
            return CommandResult(True, "好，声音调小了。", "volume_down")
        return None

    def _volume_key(self, vk, times=1):
        # 用 WScript.Shell 发音量键：不依赖任何第三方库，且不会误改其它设置
        script = ("$w = New-Object -ComObject WScript.Shell; "
                  "1..%d | ForEach-Object { $w.SendKeys([char]%d) }" % (max(1, int(times)), vk))
        return _powershell(script)

    def _h_open(self, t, p):
        m = re.search(r"(?:打开|启动|运行)(.+)", t)
        if not m:
            return None
        name = self._plain(m.group(1))
        if not name:
            return None

        # 1) 白名单里的程序
        for key, target in COMMAND_APP_WHITELIST.items():
            if key in name or name in key:
                self.last_plan = "open_app:%s->%s" % (key, target)
                if not self.dry_run:
                    try:
                        os.startfile(target)
                    except Exception as e:
                        return CommandResult(True, "打不开%s，可能没装或路径不对。" % key,
                                             "open_app_failed")
                return CommandResult(True, "好，正在打开%s。" % key, "open_app")

        # 2) 常用网站
        for key, url in COMMAND_SITE_WHITELIST.items():
            if key in name:
                self.last_plan = "open_site:%s->%s" % (key, url)
                if not self.dry_run:
                    try:
                        import webbrowser
                        webbrowser.open(url)
                    except Exception:
                        return CommandResult(True, "打不开网页，稍后再试。", "open_site_failed")
                return CommandResult(True, "好，正在打开%s。" % key, "open_site")

        # 3) 不在白名单：明确拒绝，并提示可以打开哪些（绝不执行未知名字）
        apps = "、".join(COMMAND_APP_WHITELIST.keys())
        return CommandResult(True,
                             "我只能打开这几个：%s。您说的是「%s」，我不认识，"
                             "为安全起见没有打开。" % (apps, name[:12]), "open_rejected")

    def _h_note(self, t, p):
        m = re.match(r"^(?:记一下|记一笔|帮我记|记住|记录)(.*)$", t.strip())
        if not m:
            return None
        content = m.group(1).lstrip("：: ，,、").strip()
        if not content:
            return CommandResult(True, "要记什么？说「记一下」后面跟上内容就行。", "note_empty")
        self.last_plan = "note:%s" % content[:30]
        if self.dry_run:
            return CommandResult(True, "（自检）已记录：%s" % content, "note")
        item = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "text": content}
        try:
            os.makedirs(os.path.dirname(self.notes_path), exist_ok=True)
            items = []
            if os.path.exists(self.notes_path):
                with open(self.notes_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
            items.append(item)
            with open(self.notes_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return CommandResult(True, "记不下来，写文件出错了。", "note_failed")
        return CommandResult(True, "记好了：%s" % content, "note")

    def _h_read_notes(self, t, p):
        if not any(k in p for k in ("待办", "记事", "记了", "记住了", "笔记", "我记的")):
            return None
        if any(k in p for k in ("清空", "删掉", "删除", "都删")):
            if not self.dry_run:
                try:
                    if os.path.exists(self.notes_path):
                        os.remove(self.notes_path)
                except Exception:
                    pass
            return CommandResult(True, "好，记事已经清空了。", "notes_clear")
        items = []
        try:
            if os.path.exists(self.notes_path):
                with open(self.notes_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
        except Exception:
            items = []
        if not items:
            return CommandResult(True, "您还没记过什么。说「记一下」后面跟上内容就行。",
                                 "notes_empty")
        recent = items[-5:]
        lines = "；".join("%s" % it.get("text", "") for it in recent)
        return CommandResult(True, "您最近记了%d条：%s" % (len(items), lines), "notes_read")

    def _h_clipboard(self, t, p):
        if not any(k in p for k in ("剪贴板", "复制的内容", "复制了什么")):
            return None
        out, err = _powershell("Get-Clipboard -Raw")
        if err:
            return CommandResult(True, "读不到剪贴板内容。", "clipboard_failed")
        out = (out or "").strip()
        if not out:
            return CommandResult(True, "剪贴板是空的。", "clipboard_empty")
        return CommandResult(True, "剪贴板里是：%s" % out[:200], "clipboard")

    def _h_screen_text(self, t, p):
        if not any(k in p for k in ("屏幕", "屏幕上", "读一下屏幕", "看看屏幕上")):
            return None
        if self.analyzer is None:
            return CommandResult(True, "现在读不了屏幕。", "screen_no_analyzer")
        if self.dry_run:
            self.last_plan = "screen_text"
            return CommandResult(True, "（自检）会截屏并读出屏幕上的文字", "screen_text")
        image, err = self._grab_screen()
        if err:
            return CommandResult(True, "截屏失败：%s" % err, "screen_failed")
        try:
            import asyncio
            self.analyzer.set_api_key(self.api_key)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reply, verr = loop.run_until_complete(
                self.analyzer.analyze_image(image, "读出这张屏幕截图上的主要文字内容，"
                                                   "按顺序说清楚，不要用列表符号。"))
            loop.close()
        except Exception as e:
            return CommandResult(True, "读屏幕出错了。", "screen_failed")
        if verr or not reply:
            return CommandResult(True, "屏幕上没读出什么内容。", "screen_empty")
        return CommandResult(True, reply, "screen_text")

    def _grab_screen(self):
        """截取主屏。用 Pillow（Anaconda 自带），转成 OpenCV 的 BGR 数组给视觉模块用"""
        try:
            from PIL import ImageGrab
            import numpy as np
            import cv2
            shot = ImageGrab.grab()
            arr = np.array(shot)                     # RGB
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), ""
        except Exception as e:
            return None, str(e)[:80]


if __name__ == "__main__":
    print("=== 本地指令自检（除音量/程序外都会真执行；音量与打开程序用 dry_run 只判断计划）===")
    cmds = LocalCommands(dry_run=False)
    dry = LocalCommands(dry_run=True)

    print("\n【1】匹配与计划（dry_run，不真的调音量/开程序）")
    cases = [
        "声音大一点", "声音小一点", "静音", "取消静音",
        "打开记事本", "打开计算器", "打开百度", "打开微信",
        "记一下 明天早上八点吃降压药",
        "读一下待办", "剪贴板里是什么", "现在几点", "今天几号",
        "你能做什么",
        "这个药一天吃几次",
    ]
    for c in cases:
        res = dry.try_handle(c)
        if res.handled:
            print("  %-24s -> 本地处理 [%s] 计划=%s" % (c, res.action, dry.last_plan))
            dry.last_plan = ""
        else:
            print("  %-24s -> 交给大模型" % c)

    print("\n【2】真执行的安全指令（时间 / 记事 / 读记事）")
    for c in ["现在几点", "今天几号"]:
        res = cmds.try_handle(c)
        print("  %-16s -> %s" % (c, res.reply))
    # 用临时文件测试记事，避免污染真实待办
    import tempfile
    tmp_notes = os.path.join(tempfile.mkdtemp(prefix="cmd_selftest_"), "notes.json")
    c2 = LocalCommands(notes_path=tmp_notes)
    print("  %-16s -> %s" % ("记一下 自检条目", c2.try_handle("记一下 自检条目").reply))
    print("  %-16s -> %s" % ("读一下待办", c2.try_handle("读一下待办").reply))
    print("  %-16s -> %s" % ("清空记事", c2.try_handle("清空记事").reply))
    print("  %-16s -> %s" % ("再读一次", c2.try_handle("读一下待办").reply))
    import shutil
    shutil.rmtree(os.path.dirname(tmp_notes), ignore_errors=True)

    print("\n【3】安全性：未知程序必须被拒绝（不能执行）")
    res = dry.try_handle("打开 C:\\Windows\\System32\\cmd.exe")
    print("  %s" % res.reply)
    print("  判定: %s" % ("通过（拒绝了未知目标）" if res.action == "open_rejected" else "不通过！"))

    print("\n【4】截屏能力（只检查能否取到屏幕，不调用视觉模型）")
    img, err = LocalCommands()._grab_screen()
    if err:
        print("  截屏失败：%s" % err)
    else:
        print("  截屏成功：%dx%d" % (img.shape[1], img.shape[0]))

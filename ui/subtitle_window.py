# -*- coding: utf-8 -*-
"""
大字幕面板：把对话内容用大字号显示出来，给听障、听力下降或视力不佳的用户看

为什么单独一个窗口：
  1. 需要**置顶**——对方说话时要一直看得见，不能被别的窗口挡住；
  2. 要能放到屏幕另一侧给对面的人看，所以不能塞在主界面里；
  3. 字号要能一键放大到很远都看得清。

内容分两类并配色区分：
  对方说的话（从语音转写）—— 黑色
  我们这边的回答            —— 蓝色
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QFontDialog,
)

from config import UI_SUBTITLE_FONT_SIZE, UI_HIGH_CONTRAST, SUBTITLE_MAX_ITEMS


class SubtitleWindow(QWidget):
    """大字幕窗口（置顶、可调字号、可清空）"""

    closed = pyqtSignal()

    def __init__(self, parent=None, font_size=None):
        super().__init__(None)      # 独立顶层窗口：不带 parent，才能独立置顶
        self.font_size = int(font_size or UI_SUBTITLE_FONT_SIZE)
        self._lines = []

        self.setWindowTitle("大字幕")
        flags = Qt.Window
        if UI_HIGH_CONTRAST:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.resize(760, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 顶部工具条
        bar = QHBoxLayout()
        title = QLabel("大字幕（对方看这屏）", self)
        title.setFont(QFont("Microsoft YaHei", max(12, self.font_size // 2), QFont.Bold))
        bar.addWidget(title)
        bar.addStretch(1)

        btn_bigger = QPushButton("A+", self)
        btn_bigger.setFixedWidth(56)
        btn_bigger.clicked.connect(lambda: self.change_font(+4))
        bar.addWidget(btn_bigger)

        btn_smaller = QPushButton("A-", self)
        btn_smaller.setFixedWidth(56)
        btn_smaller.clicked.connect(lambda: self.change_font(-4))
        bar.addWidget(btn_smaller)

        btn_pick = QPushButton("字号…", self)
        btn_pick.clicked.connect(self.pick_font)
        bar.addWidget(btn_pick)

        btn_clear = QPushButton("清空", self)
        btn_clear.clicked.connect(self.clear)
        bar.addWidget(btn_clear)

        btn_close = QPushButton("关闭", self)
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        layout.addLayout(bar)

        # 字幕主体
        self.view = QTextEdit(self)
        self.view.setReadOnly(True)
        self._apply_style()
        layout.addWidget(self.view, 1)

    # -------------------- 样式 --------------------

    def _apply_style(self):
        """高对比配色：默认黑底白字更适合弱视用户，对方也更容易在远处看清"""
        if UI_HIGH_CONTRAST:
            bg, fg = "#000000", "#FFFFFF"
        else:
            bg, fg = "#FFFFFF", "#1F2937"
        self.view.setStyleSheet(
            "QTextEdit { background-color: %s; color: %s; border: 2px solid #444;"
            " border-radius: 8px; padding: 12px; font-size: %dpx; }"
            % (bg, fg, self.font_size)
        )
        self.setStyleSheet("QWidget { background-color: #111111; color: #EEEEEE; }"
                           if UI_HIGH_CONTRAST else "")
        self.setFont(QFont("Microsoft YaHei", max(10, self.font_size // 2)))

    def change_font(self, delta):
        self.font_size = max(16, min(96, self.font_size + delta))
        self._apply_style()

    def pick_font(self):
        font, ok = QFontDialog.getFont(QFont("Microsoft YaHei", self.font_size), self)
        if ok:
            self.font_size = max(16, min(96, font.pointSize()))
            self._apply_style()

    # -------------------- 内容 --------------------

    def add_line(self, speaker, text, color=None):
        """
        :param speaker: "对方" / "我" / "系统"
        :param text: 内容
        """
        text = (text or "").strip()
        if not text:
            return
        self._lines.append((speaker, text))
        if len(self._lines) > SUBTITLE_MAX_ITEMS:
            self._lines = self._lines[-SUBTITLE_MAX_ITEMS:]
        if color is None:
            color = {"对方": "#FFFFFF", "我": "#7DD3FC", "系统": "#FBBF24"}.get(speaker, "#FFFFFF")
        if not UI_HIGH_CONTRAST:
            color = {"对方": "#111111", "我": "#1D4ED8", "系统": "#B45309"}.get(speaker, "#111111")
        html = ('<div style="margin-bottom:10px;">'
                '<span style="color:%s; font-weight:bold;">%s：</span>'
                '<span style="color:%s;">%s</span></div>'
                % (color, speaker, color, self._escape(text)))
        self.view.append(html)
        # 自动滚到底
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _escape(text):
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def clear(self):
        self._lines = []
        self.view.clear()

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

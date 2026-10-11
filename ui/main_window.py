# -*- coding: utf-8 -*-
"""
主窗口（无障碍版）

与旧版的三点不同：
  1. **免按键连续对话**：点一次「开始聆听」，之后靠 VAD 自动断句，不用反复点按钮；
  2. **适老化**：字号明显放大、高对比配色、按钮加大，状态用「文字 + 语音」双通道提示；
  3. **大字幕**：对方看的那一屏独立置顶显示，给听障与听力下降的用户读。

线程约定：
  DialogueEngine 的回调发生在引擎/采集线程里，**绝不能直接碰 Qt 控件**。
  这里用 EngineBridge 把它们转成 Qt 信号，由 Qt 排队到界面线程执行。
"""

import os
import sys

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QTextBrowser, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QApplication, QFrame, QVBoxLayout, QHBoxLayout,
    QCheckBox, QProgressBar, QTextEdit, QDialog, QDialogButtonBox, QSpinBox,
    QFormLayout,
)

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from core.dialogue import (
    DialogueEngine, STATE_IDLE, STATE_LISTENING, STATE_RECOGNIZING,
    STATE_THINKING, STATE_SPEAKING, STATE_LABELS,
)
from core.vision import ImageAnalyzer
from core.continuous_listener import probe as device_probe
from config import (
    API_KEY, LOCAL_VOICE_OPTIONS, DEFAULT_LOCAL_VOICE, DEFAULT_SPEECH_RATE,
    UI_FONT_SIZE, UI_SUBTITLE_FONT_SIZE, VAD_MODE, VAD_FRAME_MS, SILENCE_END_MS,
    MIN_SPEECH_MS, BARGE_IN_ENERGY_RATIO, BARGE_IN_MIN_MS, TTS_FILLERS,
    GESTURE_MODE, OBJECT_QUESTION_TEXT, OBJECT_QUESTION_SPEAK,
)


# ======================================================================
# 引擎 → 界面 的线程安全桥
# ======================================================================

class EngineBridge(QObject):
    """把 DialogueEngine / HandTracker 的回调转成 Qt 信号（回调不在界面线程里）"""
    sig_state = pyqtSignal(str)
    sig_user_text = pyqtSignal(str)
    sig_reply_chunk = pyqtSignal(str)
    sig_reply = pyqtSignal(str)
    sig_subtitle = pyqtSignal(str, str)
    sig_status = pyqtSignal(str)
    sig_error = pyqtSignal(str)
    sig_gesture = pyqtSignal(str, float)


class VisionThread(QThread):
    """视觉分析线程（摄像头/图片 → VL 模型，流式）"""
    token = pyqtSignal(str)
    done = pyqtSignal(str, str)

    def __init__(self, analyzer, image_np, question):
        super().__init__()
        self.analyzer = analyzer
        self.image_np = image_np
        self.question = question

    def run(self):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reply, err = loop.run_until_complete(
                self.analyzer.analyze_image(self.image_np, self.question,
                                            on_chunk=lambda t: self.token.emit(t)))
            loop.close()
            self.done.emit(reply, err)
        except Exception as e:
            self.done.emit("", "视觉分析异常: %s" % str(e)[:120])


class ProbeThread(QThread):
    """设备自检线程（跑探针，不阻塞界面）"""
    done = pyqtSignal(str)

    def run(self):
        import io
        import contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                device_probe(seconds=1.5)
        except Exception as e:
            buf.write("\n探针异常: %s" % e)
        self.done.emit(buf.getvalue())


# ======================================================================
# 适老化样式
# ======================================================================

def build_stylesheet(base=UI_FONT_SIZE):
    """
    适老化样式表。

    用**具名占位符**而不是位置参数：这段 QSS 里有 10 处字号，
    位置写法一旦少给一个参数就整段崩掉（刚踩过："not enough arguments"），
    具名写法既看得清对应关系，也不会因为顺序调整出错。
    """
    return """
QWidget#MainWindow { background-color: #F2F5FA; }
QFrame#banner {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1D4ED8, stop:1 #0F766E);
}
QLabel#bannerTitle { color: #FFFFFF; font-size: %(banner)dpx; font-weight: bold; letter-spacing: 2px; }
QFrame#card { background-color: #FFFFFF; border: 2px solid #C7D2E1; border-radius: 14px; }
QLabel#cardTitle { color: #0F172A; font-size: %(card)dpx; font-weight: bold; }
QLabel#hint { color: #475569; font-size: %(hint)dpx; }
QLabel#stateLabel { color: #0F172A; font-size: %(state)dpx; font-weight: bold; }
QTextBrowser, QTextEdit {
    background-color: #FFFFFF; border: 2px solid #C7D2E1; border-radius: 10px;
    padding: 10px; font-size: %(text)dpx; color: #0F172A;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #FFFFFF; border: 2px solid #C7D2E1; border-radius: 9px;
    padding: 6px 10px; font-size: %(input)dpx; color: #0F172A;
}
QLineEdit:focus, QComboBox:focus { border: 3px solid #1D4ED8; }
QPushButton {
    background-color: #FFFFFF; color: #1D4ED8; border: 2px solid #1D4ED8;
    border-radius: 10px; font-size: %(button)dpx; padding: 8px 14px;
}
QPushButton:hover { background-color: #E8F0FE; }
QPushButton:disabled { color: #94A3B8; border-color: #CBD5E1; background-color: #F1F5F9; }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1D4ED8, stop:1 #0F766E);
    color: #FFFFFF; border: none; font-weight: bold; font-size: %(primary)dpx;
}
QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #2563EB, stop:1 #0E7490); }
QPushButton#danger { color: #B91C1C; border-color: #DC2626; background-color: #FEF2F2; }
QPushButton#danger:hover { background-color: #FEE2E2; }
QProgressBar { border: 2px solid #C7D2E1; border-radius: 8px; background: #FFFFFF;
    height: 18px; text-align: center; font-size: %(bar)dpx; }
QProgressBar::chunk { background-color: #16A34A; border-radius: 6px; }
QCheckBox { font-size: %(check)dpx; color: #0F172A; }
""" % {
        "banner": base + 8,
        "card": base + 2,
        "hint": max(11, base - 3),
        "state": base + 8,
        "text": base,
        "input": base,
        "button": base,
        "primary": base + 4,
        "bar": max(11, base - 3),
        "check": base - 1,
    }


# ======================================================================
# 主窗口
# ======================================================================

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.engine = None
        self.bridge = EngineBridge()
        self.bridge.sig_state.connect(self.on_state)
        self.bridge.sig_user_text.connect(self.on_user_text)
        self.bridge.sig_reply_chunk.connect(self.on_reply_chunk)
        self.bridge.sig_reply.connect(self.on_reply)
        self.bridge.sig_subtitle.connect(self.on_subtitle)
        self.bridge.sig_status.connect(self.on_status)
        self.bridge.sig_error.connect(self.on_error)
        self.bridge.sig_gesture.connect(self.on_gesture)

        self.analyzer = ImageAnalyzer()
        self.current_image = None
        self._display_buf = None
        self._threads = []
        self.subtitle_window = None
        self.tracker = None          # 手势识别（HandTracker）
        self._reply_open = False     # 是否正在流式渲染一条回复（决定要不要先插「助手：」）

        self.font_size = UI_FONT_SIZE
        self.init_ui()

        # 电平条刷新（读的是采集线程写的统计值，只读不写，安全）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_level)
        self.timer.start(120)

    # -------------------- 界面 --------------------

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("智能语音助手 · 无障碍版")
        self.resize(1180, 800)
        self.setMinimumSize(1000, 680)
        self.setStyleSheet(build_stylesheet(self.font_size))
        self.setFont(QFont("Microsoft YaHei", self.font_size - 4))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        banner = QFrame(self)
        banner.setObjectName("banner")
        banner.setFixedHeight(72)
        bl = QVBoxLayout(banner)
        bl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("智能语音助手 · 无障碍版", banner)
        title.setObjectName("bannerTitle")
        title.setAlignment(Qt.AlignCenter)
        bl.addWidget(title)
        outer.addWidget(banner)

        body = QHBoxLayout()
        body.setContentsMargins(18, 14, 18, 18)
        body.setSpacing(16)
        outer.addLayout(body, 1)

        # ================= 左栏 =================
        left = QVBoxLayout()
        left.setSpacing(14)
        body.addLayout(left, 6)

        # ---- 状态卡片（核心操作）----
        card_state = QFrame(self)
        card_state.setObjectName("card")
        sv = QVBoxLayout(card_state)
        sv.setContentsMargins(18, 14, 18, 16)
        sv.setSpacing(10)

        lbl = QLabel("语音对话", card_state)
        lbl.setObjectName("cardTitle")
        sv.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.btn_listen = QPushButton("开始聆听", card_state)
        self.btn_listen.setObjectName("primary")
        self.btn_listen.setMinimumSize(220, 64)
        self.btn_listen.clicked.connect(self.on_listen_toggle)
        row.addWidget(self.btn_listen)

        right_box = QVBoxLayout()
        right_box.setSpacing(8)
        self.lbl_state = QLabel("状态：空闲", card_state)
        self.lbl_state.setObjectName("stateLabel")
        right_box.addWidget(self.lbl_state)

        self.level = QProgressBar(card_state)
        self.level.setRange(0, 100)
        self.level.setValue(0)
        self.level.setFormat("麦克风音量")
        right_box.addWidget(self.level)
        row.addLayout(right_box, 1)
        sv.addLayout(row)

        self.lbl_hint = QLabel(
            "点一次「开始聆听」后就不需要再点按钮：说完停顿会自动识别，"
            "回答播报时你可以随时开口打断。", card_state)
        self.lbl_hint.setObjectName("hint")
        self.lbl_hint.setWordWrap(True)
        sv.addWidget(self.lbl_hint)
        left.addWidget(card_state)

        # ---- 对话记录 ----
        card_chat = QFrame(self)
        card_chat.setObjectName("card")
        cv = QVBoxLayout(card_chat)
        cv.setContentsMargins(18, 14, 18, 16)
        cv.setSpacing(8)
        lbl_chat = QLabel("对话记录", card_chat)
        lbl_chat.setObjectName("cardTitle")
        cv.addWidget(lbl_chat)
        self.chat = QTextBrowser(card_chat)
        self.chat.setMinimumHeight(240)
        self.chat.append("<span style='color:#475569;'>准备就绪。点「开始聆听」后直接说话即可。</span>")
        cv.addWidget(self.chat, 1)
        left.addWidget(card_chat, 1)

        # ---- 文字输入（备用通道）----
        card_text = QFrame(self)
        card_text.setObjectName("card")
        tv = QVBoxLayout(card_text)
        tv.setContentsMargins(18, 14, 18, 16)
        tv.setSpacing(8)
        lbl_text = QLabel("打字输入（没有麦克风时也能用）", card_text)
        lbl_text.setObjectName("cardTitle")
        tv.addWidget(lbl_text)
        th = QHBoxLayout()
        th.setSpacing(10)
        self.input_text = QLineEdit(card_text)
        self.input_text.setMinimumHeight(46)
        self.input_text.setPlaceholderText("输入内容后按回车…")
        self.input_text.returnPressed.connect(self.on_text_send)
        th.addWidget(self.input_text, 1)
        self.btn_send = QPushButton("发送", card_text)
        self.btn_send.setObjectName("primary")
        self.btn_send.setMinimumSize(110, 46)
        self.btn_send.clicked.connect(self.on_text_send)
        th.addWidget(self.btn_send)
        tv.addLayout(th)
        left.addWidget(card_text)

        # ================= 右栏 =================
        right = QVBoxLayout()
        right.setSpacing(14)
        body.addLayout(right, 5)

        # ---- 大字幕 ----
        card_sub = QFrame(self)
        card_sub.setObjectName("card")
        subv = QVBoxLayout(card_sub)
        subv.setContentsMargins(18, 14, 18, 16)
        subv.setSpacing(8)
        lbl_sub = QLabel("大字幕（给听障／听力下降的人看）", card_sub)
        lbl_sub.setObjectName("cardTitle")
        subv.addWidget(lbl_sub)
        sh = QHBoxLayout()
        sh.setSpacing(10)
        self.btn_subtitle = QPushButton("打开大字幕窗口", card_sub)
        self.btn_subtitle.setMinimumHeight(46)
        self.btn_subtitle.clicked.connect(self.on_toggle_subtitle)
        sh.addWidget(self.btn_subtitle, 1)
        subv.addLayout(sh)
        note_sub = QLabel("字幕窗口默认置顶，可以拖到屏幕另一侧给对方看，"
                          "字号用窗口里的 A+ / A- 调整。", card_sub)
        note_sub.setObjectName("hint")
        note_sub.setWordWrap(True)
        subv.addWidget(note_sub)
        right.addWidget(card_sub)

        # ---- 手势识别 ----
        card_gest = QFrame(self)
        card_gest.setObjectName("card")
        gv = QVBoxLayout(card_gest)
        gv.setContentsMargins(18, 14, 18, 16)
        gv.setSpacing(8)
        lbl_gest = QLabel("手势识别（受限词表，非标准手语）", card_gest)
        lbl_gest.setObjectName("cardTitle")
        gv.addWidget(lbl_gest)

        gh = QHBoxLayout()
        gh.setSpacing(10)
        self.btn_gesture = QPushButton("开启手势识别", card_gest)
        self.btn_gesture.setMinimumHeight(46)
        self.btn_gesture.clicked.connect(self.on_gesture_toggle)
        gh.addWidget(self.btn_gesture, 1)

        self.combo_gesture = QComboBox(card_gest)
        self.combo_gesture.addItem("手势用于控制", "control")
        self.combo_gesture.addItem("手势用于说话", "speak")
        self.combo_gesture.setMinimumHeight(46)
        self.combo_gesture.currentIndexChanged.connect(self.on_gesture_mode_changed)
        gh.addWidget(self.combo_gesture, 1)
        gv.addLayout(gh)

        self.lbl_gesture = QLabel("最近手势：—", card_gest)
        self.lbl_gesture.setObjectName("hint")
        gv.addWidget(self.lbl_gesture)

        note_gest = QLabel(
            "控制模式：✋开始/停止聆听、✊取消、👍确认、👎停止播报、✌️切字幕、"
            "挥手紧急停止、☝️指物提问。\n"
            "说话模式：把手势读成短语说给对方听（面向聋哑用户，词表见说明）。",
            card_gest)
        note_gest.setObjectName("hint")
        note_gest.setWordWrap(True)
        gv.addWidget(note_gest)
        right.addWidget(card_gest)

        # ---- 视觉（为手语/指物保留）----
        card_vision = QFrame(self)
        card_vision.setObjectName("card")
        vv = QVBoxLayout(card_vision)
        vv.setContentsMargins(18, 14, 18, 16)
        vv.setSpacing(8)
        lbl_vis = QLabel("拍照识别（手语／指物问答将基于此）", card_vision)
        lbl_vis.setObjectName("cardTitle")
        vv.addWidget(lbl_vis)
        self.lbl_image = QLabel("（此处显示图片）", card_vision)
        self.lbl_image.setMinimumSize(260, 150)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 2px dashed #94A3B8; background:#F8FAFC;"
                                     "color:#64748B; border-radius:10px;")
        vv.addWidget(self.lbl_image, 1)
        vh = QHBoxLayout()
        self.btn_camera = QPushButton("拍照", card_vision)
        self.btn_camera.setMinimumHeight(42)
        self.btn_camera.clicked.connect(self.on_capture)
        vh.addWidget(self.btn_camera, 1)
        self.btn_pick = QPushButton("选择图片", card_vision)
        self.btn_pick.setMinimumHeight(42)
        self.btn_pick.clicked.connect(self.on_pick_image)
        vh.addWidget(self.btn_pick, 1)
        vv.addLayout(vh)
        qh = QHBoxLayout()
        self.input_image_q = QLineEdit(card_vision)
        self.input_image_q.setMinimumHeight(40)
        self.input_image_q.setPlaceholderText("对图片的问题（可选）")
        qh.addWidget(self.input_image_q, 1)
        self.btn_analyze = QPushButton("识别", card_vision)
        self.btn_analyze.setObjectName("primary")
        self.btn_analyze.setMinimumSize(90, 40)
        self.btn_analyze.clicked.connect(self.on_analyze)
        qh.addWidget(self.btn_analyze)
        vv.addLayout(qh)

        self.check_speak_image = QCheckBox("识别结果语音报出（指物问答）", card_vision)
        self.check_speak_image.setChecked(OBJECT_QUESTION_SPEAK)
        vv.addWidget(self.check_speak_image)
        right.addWidget(card_vision, 1)

        # ---- 设置 ----
        card_set = QFrame(self)
        card_set.setObjectName("card")
        setv = QVBoxLayout(card_set)
        setv.setContentsMargins(18, 14, 18, 16)
        setv.setSpacing(8)
        lbl_set = QLabel("设置", card_set)
        lbl_set.setObjectName("cardTitle")
        setv.addWidget(lbl_set)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.input_api = QLineEdit(card_set)
        self.input_api.setEchoMode(QLineEdit.Password)
        self.input_api.setPlaceholderText("硅基流动 API Key")
        self.input_api.setMinimumHeight(38)
        if API_KEY:
            self.input_api.setText(API_KEY)
        form.addRow("API Key：", self.input_api)

        self.combo_voice = QComboBox(card_set)
        for vid, desc in LOCAL_VOICE_OPTIONS:
            self.combo_voice.addItem(desc, vid)
        idx = self.combo_voice.findData(DEFAULT_LOCAL_VOICE)
        if idx >= 0:
            self.combo_voice.setCurrentIndex(idx)
        self.combo_voice.setMinimumHeight(38)
        form.addRow("音色：", self.combo_voice)

        self.combo_rate = QComboBox(card_set)
        for r, label in (("-20%", "很慢"), ("-10%", "偏慢（推荐）"), ("+0%", "正常"), ("+10%", "偏快")):
            self.combo_rate.addItem(label, r)
        idx = self.combo_rate.findData(DEFAULT_SPEECH_RATE)
        if idx >= 0:
            self.combo_rate.setCurrentIndex(idx)
        self.combo_rate.setMinimumHeight(38)
        form.addRow("语速：", self.combo_rate)

        self.spin_font = QSpinBox(card_set)
        self.spin_font.setRange(12, 32)
        self.spin_font.setValue(self.font_size)
        self.spin_font.setMinimumHeight(38)
        self.spin_font.valueChanged.connect(self.on_font_changed)
        form.addRow("界面字号：", self.spin_font)

        self.check_search = QCheckBox("回答前先联网检索", card_set)
        form.addRow("联网：", self.check_search)
        setv.addLayout(form)

        btns = QHBoxLayout()
        self.btn_probe = QPushButton("设备自检", card_set)
        self.btn_probe.setMinimumHeight(42)
        self.btn_probe.clicked.connect(self.on_device_probe)
        btns.addWidget(self.btn_probe, 1)
        self.btn_params = QPushButton("参数说明", card_set)
        self.btn_params.setMinimumHeight(42)
        self.btn_params.clicked.connect(self.on_show_params)
        btns.addWidget(self.btn_params, 1)
        setv.addLayout(btns)

        self.btn_quit = QPushButton("退出程序", card_set)
        self.btn_quit.setObjectName("danger")
        self.btn_quit.setMinimumHeight(42)
        self.btn_quit.clicked.connect(self.close)
        setv.addWidget(self.btn_quit)
        right.addWidget(card_set)

    # -------------------- 引擎 --------------------

    def _api_key(self):
        key = self.input_api.text().strip()
        return key if key else API_KEY

    def _ensure_engine(self, open_listener=True):
        """按需创建并启动引擎（首次点「开始聆听」或发送文字时）"""
        if self.engine is not None and self.engine.is_running():
            return True, ""
        if not self._api_key():
            QMessageBox.warning(self, "提示",
                                "请先在「设置」里填入硅基流动 API Key。\n"
                                "注册地址：https://siliconflow.cn")
            return False, "缺少 API Key"

        self.engine = DialogueEngine(
            on_state=self.bridge.sig_state.emit,
            on_user_text=self.bridge.sig_user_text.emit,
            on_reply_chunk=self.bridge.sig_reply_chunk.emit,
            on_reply=self.bridge.sig_reply.emit,
            on_subtitle=self.bridge.sig_subtitle.emit,
            on_status=self.bridge.sig_status.emit,
            on_error=self.bridge.sig_error.emit,
            api_key=self._api_key(),
            voice=self.combo_voice.currentData(),
            rate=self.combo_rate.currentData(),
        )
        self.engine.enable_web_search = self.check_search.isChecked()
        ok, err = self.engine.start(open_listener=open_listener)
        if not ok:
            self.engine = None
            QMessageBox.critical(self, "启动失败",
                                 "无法打开音频设备：\n%s\n\n可以先只用打字输入。" % err)
            return False, err
        return True, ""

    # -------------------- 槽：状态与消息 --------------------

    def on_listen_toggle(self):
        if (self.engine is not None and self.engine.is_running()
                and self.engine.listener is not None):
            self.engine.stop()
            self.engine = None
            self.btn_listen.setText("开始聆听")
            self.lbl_state.setText("状态：已停止")
            self.chat.append("<span style='color:#B45309;'>已停止聆听。</span>")
            return

        ok, err = self._ensure_engine(open_listener=True)
        if not ok:
            return
        self.btn_listen.setText("停止聆听")
        self.chat.append("<span style='color:#0F766E;'>开始聆听，请直接说话（不用按键）。</span>")

    def on_state(self, state):
        label = STATE_LABELS.get(state, state)
        self.lbl_state.setText("状态：%s" % label)
        colors = {STATE_LISTENING: "#0F766E", STATE_RECOGNIZING: "#B45309",
                  STATE_THINKING: "#1D4ED8", STATE_SPEAKING: "#7C3AED"}
        color = colors.get(state, "#0F172A")
        self.lbl_state.setStyleSheet("color:%s; font-weight:bold;" % color)

    def on_user_text(self, text):
        self._reply_open = False
        self.chat.append("<b style='color:#1D4ED8;'>用户：</b>%s" % self._esc(text))

    def on_reply_chunk(self, token):
        """
        流式渲染回复。

        必须**先插一行「助手：」再往里追加**：改造时一度去掉了这一步，
        结果回答被直接粘在用户那句话后面，显示成
        「用户：你好…能听见您说话…」（这类问题只有整体联调才暴露得出来）。
        """
        if not self._reply_open:
            self.chat.append("<b style='color:#0F766E;'>助手：</b>")
            self._reply_open = True
        self.chat.moveCursor(self.chat.textCursor().End)
        self.chat.insertPlainText(token)

    def on_reply(self, reply):
        """
        一条回复结束。
        注意：如果这次没有走过 token 回调（例如本地指令是整段返回的），
        回复内容只在这里出现，必须补渲染，否则用户看不到任何回答。
        """
        if not self._reply_open and reply:
            self.chat.append("<b style='color:#0F766E;'>助手：</b>%s" % self._esc(reply))
        self._reply_open = False
        self.chat.append("")
        self.chat.moveCursor(self.chat.textCursor().End)

    def on_subtitle(self, speaker, text):
        if self.subtitle_window is not None and self.subtitle_window.isVisible():
            self.subtitle_window.add_line(speaker, text)

    def on_status(self, text):
        self.chat.append("<i style='color:#475569;'>%s</i>" % self._esc(text))

    def on_error(self, text):
        self.chat.append("<i style='color:#B91C1C;'>%s</i>" % self._esc(text))

    @staticmethod
    def _esc(text):
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def update_level(self):
        if self.engine is None or self.engine.listener is None:
            self.level.setValue(0)
            return
        level = self.engine.listener.get_level()
        # 麦克风 RMS 通常在 0~0.3；映射到 0~100 并做上限保护
        self.level.setValue(int(max(0.0, min(1.0, level / 0.30)) * 100))

    # -------------------- 打字输入 --------------------

    def on_text_send(self):
        text = self.input_text.text().strip()
        if not text:
            return
        ok, err = self._ensure_engine(open_listener=False)
        if not ok:
            return
        self.engine.submit_text(text)
        self.input_text.clear()

    # -------------------- 手势识别 --------------------

    def on_gesture_toggle(self):
        if self.tracker is not None and self.tracker.is_running():
            self.tracker.stop()
            self.tracker = None
            self.btn_gesture.setText("开启手势识别")
            self.lbl_gesture.setText("最近手势：—")
            self.on_status("手势识别已停止（摄像头已释放）")
            return

        from core.gesture import HandTracker
        self.tracker = HandTracker(
            on_gesture=self.bridge.sig_gesture.emit,
            on_status=self.bridge.sig_status.emit,
            on_error=self.bridge.sig_error.emit,
            mode=self.combo_gesture.currentData() or GESTURE_MODE,
        )
        ok, err = self.tracker.start()
        if not ok:
            self.tracker = None
            QMessageBox.warning(self, "手势识别无法启动", err)
            return
        self.btn_gesture.setText("停止手势识别")
        self.on_status("手势识别已启动，模式：%s"
                       % ("控制" if self.tracker.mode == "control" else "说话"))

    def on_gesture_mode_changed(self, index):
        if self.tracker is not None:
            self.tracker.set_mode(self.combo_gesture.currentData())
            self.on_status("手势模式切换为：%s"
                           % ("控制程序" if self.tracker.mode == "control" else "说话（读出短语）"))

    def on_gesture(self, name, confidence):
        """手势事件（已在界面线程）"""
        from core.gesture import HandTracker
        label = name
        self.lbl_gesture.setText("最近手势：%s（置信度 %.2f）" % (label, confidence))
        self.chat.append("<i style='color:#7C3AED;'>[手势] %s</i>" % self._esc(label))

        mode = self.tracker.mode if self.tracker is not None else GESTURE_MODE
        if mode == "speak":
            phrase = HandTracker.phrase_for(name)
            if phrase:
                self.chat.append("<b style='color:#0F766E;'>（手势说话）</b>%s" % self._esc(phrase))
                self.on_subtitle("我", phrase)
                self._speak(phrase)
            else:
                self.on_status("手势「%s」不在说话词表里" % label)
            return

        action = HandTracker.action_for(name)
        if not action:
            self.on_status("手势「%s」在控制模式里没有绑定动作" % label)
            return
        self.on_status("手势触发：%s" % HandTracker.action_label(action))
        self._dispatch_gesture(action)

    def _dispatch_gesture(self, action):
        if action == "toggle_listening":
            self.on_listen_toggle()
        elif action == "cancel":
            self.input_text.clear()
            self.on_status("已取消当前输入")
        elif action == "confirm":
            if self.input_text.text().strip():
                self.on_text_send()
            else:
                self.on_status("输入框是空的，没有可确认的内容")
        elif action == "stop_speaking":
            if self.engine is not None and self.engine.player is not None:
                self.engine.player.interrupt()
                self.on_status("已停止播报")
        elif action == "toggle_subtitle":
            self.on_toggle_subtitle()
        elif action == "object_question":
            self._object_question()
        elif action == "emergency_stop":
            self._emergency_stop()

    def _emergency_stop(self):
        """紧急停止：立刻断开麦克风与播报（隐私优先，动作要干脆）"""
        try:
            if self.engine is not None:
                if self.engine.player is not None:
                    self.engine.player.interrupt()
                self.engine.stop()
        except Exception:
            pass
        self.engine = None
        self.btn_listen.setText("开始聆听")
        self.lbl_state.setText("状态：已紧急停止")
        self.on_status("已紧急停止：麦克风已断开")
        self.on_subtitle("系统", "已紧急停止（麦克风断开）")

    def _speak(self, text):
        """直接读出一句话（手势短语、指物问答结果）"""
        ok, err = self._ensure_engine(open_listener=False)
        if not ok:
            return
        self.engine.speak_text(text)

    def _object_question(self):
        """指物提问：拍一张 → 问"这是什么" → 语音报出结果"""
        image, err = self.analyzer.capture_from_camera()
        if err:
            QMessageBox.warning(self, "拍照失败", err)
            return
        self.current_image = image
        self._show_image(image)
        self.input_image_q.setText(OBJECT_QUESTION_TEXT)
        self.on_analyze()

    # -------------------- 大字幕 --------------------

    def on_toggle_subtitle(self):
        if self.subtitle_window is not None and self.subtitle_window.isVisible():
            self.subtitle_window.close()
            self.subtitle_window = None
            self.btn_subtitle.setText("打开大字幕窗口")
            return
        from ui.subtitle_window import SubtitleWindow
        self.subtitle_window = SubtitleWindow(font_size=UI_SUBTITLE_FONT_SIZE)
        self.subtitle_window.show()
        self.btn_subtitle.setText("关闭大字幕窗口")
        self.subtitle_window.add_line("系统", "大字幕已开启，对方说的话会显示在这里。")

    # -------------------- 视觉 --------------------

    def on_capture(self):
        image, err = self.analyzer.capture_from_camera()
        if err:
            QMessageBox.warning(self, "拍照失败", err)
            return
        self.current_image = image
        self._show_image(image)

    def on_pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                              "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        image, err = self.analyzer.load_from_file(path)
        if err:
            QMessageBox.warning(self, "读取失败", err)
            return
        self.current_image = image
        self._show_image(image)

    def _show_image(self, image_np):
        try:
            rgb = np.ascontiguousarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            h, w, ch = rgb.shape
            self._display_buf = rgb.tobytes()
            qimg = QImage(self._display_buf, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            target = self.lbl_image.size()
            self.lbl_image.setPixmap(pix.scaled(max(target.width(), 1),
                                                max(target.height(), 1),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.lbl_image.setText("图片显示失败: %s" % str(e)[:40])

    def on_analyze(self):
        if self.current_image is None:
            QMessageBox.information(self, "提示", "请先拍照或选择一张图片")
            return
        if not self._api_key():
            QMessageBox.warning(self, "提示", "请先填入 API Key")
            return
        question = self.input_image_q.text().strip() or "请描述这张图片"
        self.analyzer.set_api_key(self._api_key())
        self.chat.append("<b style='color:#1D4ED8;'>[图片问题]</b>%s" % self._esc(question))
        self.chat.append("<b style='color:#0F766E;'>AI：</b>")
        self.on_subtitle("系统", "[图片] " + question)

        def on_token(tok):
            self.chat.moveCursor(self.chat.textCursor().End)
            self.chat.insertPlainText(tok)

        def on_done(reply, err):
            if err:
                self.chat.append("<i style='color:#B91C1C;'>%s</i>" % self._esc(err))
                return
            self.on_subtitle("我", reply)
            # 指物问答：把结果读出来（老年用户通常不想读屏幕上的字）
            if reply and self.check_speak_image.isChecked():
                self._speak(reply)

        th = VisionThread(self.analyzer, self.current_image.copy(), question)
        th.token.connect(on_token)
        th.done.connect(on_done)
        self._threads.append(th)

        def cleanup():
            if th in self._threads:
                self._threads.remove(th)

        th.finished.connect(cleanup)
        th.start()

    # -------------------- 设置 --------------------

    def on_font_changed(self, value):
        self.font_size = int(value)
        self.setStyleSheet(build_stylesheet(self.font_size))
        self.setFont(QFont("Microsoft YaHei", self.font_size - 4))

    def on_device_probe(self):
        self.btn_probe.setEnabled(False)
        self.btn_probe.setText("自检中…")

        def done(text):
            self.btn_probe.setEnabled(True)
            self.btn_probe.setText("设备自检")
            dlg = QDialog(self)
            dlg.setWindowTitle("设备自检结果")
            dlg.resize(780, 540)
            v = QVBoxLayout(dlg)
            view = QTextEdit(dlg)
            view.setReadOnly(True)
            view.setPlainText(text)
            v.addWidget(view)
            box = QDialogButtonBox(QDialogButtonBox.Close, dlg)
            box.button(QDialogButtonBox.Close).setText("关闭")
            box.rejected.connect(dlg.reject)
            box.accepted.connect(dlg.accept)
            v.addWidget(box)
            dlg.exec_()

        th = ProbeThread()
        th.done.connect(done)
        self._threads.append(th)
        th.start()

    def on_show_params(self):
        lines = [
            "当前生效的语音参数",
            "（改动请编辑项目里的 config.py，改完重启程序）",
            "",
            "【断句】webrtcvad 模式 %d ｜ 帧长 %d ms ｜ 说完静音 %d ms 算一句结束"
            % (VAD_MODE, VAD_FRAME_MS, SILENCE_END_MS),
            "        少于 %d ms 的语音判为噪声丢弃 ｜ 单句最长 20 秒强制切分" % MIN_SPEECH_MS,
            "",
            "【打断】能量需高于本底 %.1f 倍，且持续 %d ms 才算插话（另有 300ms 起播保护期）"
            % (BARGE_IN_ENERGY_RATIO, BARGE_IN_MIN_MS),
            "",
            "【即时应答音】%d 句：%s" % (len(TTS_FILLERS), "、".join(TTS_FILLERS)),
            "",
            "为什么这些参数写在文件里而不做在界面上：",
            "断句与打断参数在采集线程启动时就已生效，界面改了不会即时生效，",
            "放进界面反而会让人以为改了却没变。所以这里只做透明展示。",
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("参数说明")
        dlg.resize(740, 440)
        v = QVBoxLayout(dlg)
        view = QTextEdit(dlg)
        view.setReadOnly(True)
        view.setPlainText("\n".join(lines))
        v.addWidget(view)
        box = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        box.button(QDialogButtonBox.Close).setText("关闭")
        box.rejected.connect(dlg.reject)
        box.accepted.connect(dlg.accept)
        v.addWidget(box)
        dlg.exec_()

    # -------------------- 退出 --------------------

    def closeEvent(self, event):
        try:
            if self.tracker is not None:
                self.tracker.stop()      # 释放摄像头
        except Exception:
            pass
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            pass
        try:
            if self.subtitle_window is not None:
                self.subtitle_window.close()
        except Exception:
            pass
        event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

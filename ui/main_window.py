# -*- coding: utf-8 -*-
"""
GUI 主窗口 + QThread 工作线程
采用 PyQt5 布局管理器（Layout）实现响应式界面：
  - 窗口放大/缩小时，对话区、图片区自动跟随拉伸
  - 字号已整体加大，高分屏下更清晰
所有耗时操作都在 QThread 子线程中执行，通过 pyqtSignal 回传结果
"""

import sys
import os
import asyncio

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QTextBrowser, QLineEdit,
    QComboBox, QFileDialog, QMessageBox, QApplication, QFrame,
    QVBoxLayout, QHBoxLayout, QCheckBox,
)

# 确保能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from core.recorder import Recorder
from core.stt import SoundToText
from core.llm import LLM
from core.tts import TextToSound
from core.player import AudioPlayer
from core.vision import ImageAnalyzer
from core.searcher import WebSearch
from config import (
    API_KEY, DEFAULT_LOCAL_VOICE, LOCAL_VOICE_OPTIONS,
    DEFAULT_TTS_PROVIDER, TTS_LOCAL_SERVICE, TTS_SILICONFLOW,
    TTS_TIMEOUT_SECONDS, TTS_MAX_TEXT_CHARS,
)


# ======================================================================
# 第一部分：QThread 工作线程类
# ======================================================================

class LLMStreamThread(QThread):
    """LLM 流式对话线程
    信号：
        token_received(str)  - 每收到一个 token 触发
        done(str, str)       - 完整回复 (回复文本, 错误信息)
    """
    token_received = pyqtSignal(str)
    done = pyqtSignal(str, str)

    def __init__(self, llm_instance, text, context="", concise_hint=None):
        super().__init__()
        self.llm = llm_instance
        self.text = text
        self.context = context
        self.concise_hint = concise_hint

    def run(self):
        reply, err = self.llm.chat_stream(
            self.text,
            on_chunk=lambda t: self.token_received.emit(t),
            context=self.context,
            concise_hint=self.concise_hint,
        )
        self.done.emit(reply, err)


class ASRThread(QThread):
    """语音识别（ASR）线程
    信号：
        done(str, str)  - (识别文本, 错误信息)
    """
    done = pyqtSignal(str, str)

    def __init__(self, stt_instance, audio_buffer):
        super().__init__()
        self.stt = stt_instance
        self.audio_buffer = audio_buffer

    def run(self):
        text, err = self.stt.transcribe(self.audio_buffer)
        self.done.emit(text, err)


class TTSThread(QThread):
    """语音合成（TTS）线程
    edge-tts / CosyVoice2 都是异步 API，需要在子线程里 asyncio.run()
    带硬超时：网络卡住时最多等 TTS_TIMEOUT_SECONDS 秒，超时自动降级为文字回复
    信号：
        done(bytes, str)  - (MP3 字节, 错误信息)
    """
    done = pyqtSignal(bytes, str)

    def __init__(self, tts_instance, text):
        super().__init__()
        self.tts = tts_instance
        self.text = text

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_bytes = loop.run_until_complete(
                asyncio.wait_for(
                    self.tts.run(self.text),
                    timeout=TTS_TIMEOUT_SECONDS,
                )
            )
            loop.close()

            if audio_bytes:
                self.done.emit(audio_bytes, "")
            else:
                self.done.emit(b"", "语音合成失败")
        except asyncio.TimeoutError:
            self.done.emit(
                b"", f"语音合成超时（超过 {TTS_TIMEOUT_SECONDS} 秒），已自动降级为文字回复"
            )
        except Exception as e:
            self.done.emit(b"", f"TTS 异常: {str(e)}")


class VisionThread(QThread):
    """视觉多模态识别线程（异步 API）
    信号：
        token_received(str)  - 流式 token
        done(str, str)       - (完整回复, 错误信息)
    """
    token_received = pyqtSignal(str)
    done = pyqtSignal(str, str)

    def __init__(self, analyzer, image_np, question):
        super().__init__()
        self.analyzer = analyzer
        self.image_np = image_np
        self.question = question

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reply, err = loop.run_until_complete(
                self.analyzer.analyze_image(
                    self.image_np,
                    self.question,
                    on_chunk=lambda t: self.token_received.emit(t),
                )
            )
            loop.close()
            self.done.emit(reply, err)
        except Exception as e:
            self.done.emit("", f"视觉分析异常: {str(e)}")


class SearchThread(QThread):
    """联网搜索线程（纯标准库实现，无需 API Key）
    信号：
        done(list, str)  - (搜索结果列表, 错误信息)
    """
    done = pyqtSignal(list, str)

    def __init__(self, query):
        super().__init__()
        self.query = query
        self.searcher = WebSearch()

    def run(self):
        results, err = self.searcher.search(self.query)
        self.done.emit(results, err)


# ======================================================================
# 第二部分：主窗口 MainWindow
# ======================================================================

# 全局 QSS 样式表（卡片式、现代浅色主题，字号已加大）
GLOBAL_QSS = """
QWidget#MainWindow {
    background-color: #EEF1F6;
}
QFrame#banner {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #2B7DE9, stop:1 #00B4A6);
    border: none;
}
QLabel#bannerTitle {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 3px;
}
QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #E1E6EF;
    border-radius: 14px;
}
QLabel#cardTitle {
    color: #344054;
    font-size: 15px;
    font-weight: bold;
}
QLabel#hintText {
    color: #9AA3B2;
    font-size: 12px;
}
QTextBrowser {
    background-color: #FAFBFC;
    border: 1px solid #E1E6EF;
    border-radius: 10px;
    padding: 10px;
    font-size: 14px;
    color: #1F2937;
}
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #D5DCE8;
    border-radius: 9px;
    padding: 4px 12px;
    font-size: 14px;
    color: #1F2937;
}
QLineEdit:focus {
    border: 2px solid #2B7DE9;
}
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D5DCE8;
    border-radius: 9px;
    padding: 4px 10px;
    font-size: 14px;
    color: #1F2937;
}
QComboBox:focus {
    border: 2px solid #2B7DE9;
}
QComboBox QAbstractItemView {
    border: 1px solid #D5DCE8;
    background: #FFFFFF;
    selection-background-color: #EBF1FE;
    selection-color: #1F2937;
    font-size: 14px;
}
QPushButton {
    background-color: #FFFFFF;
    color: #2B7DE9;
    border: 1px solid #2B7DE9;
    border-radius: 9px;
    font-size: 14px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: #EBF1FE; }
QPushButton:pressed { background-color: #D9E6FD; }
QPushButton:disabled {
    color: #9AA3B2;
    border-color: #D5DCE8;
    background-color: #F2F4F8;
}
QPushButton#sendBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #2B7DE9, stop:1 #1E6BD6);
    color: #FFFFFF;
    border: none;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#sendBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #3579E0, stop:1 #245FBE);
}
QPushButton#quitBtn {
    background-color: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FECACA;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#quitBtn:hover { background-color: #FEE2E2; }
"""

# 录音按钮两种状态样式
RECORD_BTN_IDLE = """
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #2B7DE9, stop:1 #1E6BD6);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #3579E0, stop:1 #245FBE);
}
"""
RECORD_BTN_RECORDING = """
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #EF4444, stop:1 #DC2626);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #F87171, stop:1 #EF4444);
}
"""


class MainWindow(QWidget):
    """多模态智能语音助手主窗口（响应式布局）"""

    # 各个状态文字
    STATE_IDLE = "空闲"
    STATE_RECORDING = "录音中"
    STATE_ASR = "语音识别中"
    STATE_LLM = "大模型回复中"
    STATE_TTS = "语音合成"
    STATE_PLAYING = "语音播报"

    # 播放完成信号（从播放监控线程安全回到主线程）
    sig_playback_done = pyqtSignal()

    def __init__(self):
        super().__init__()
        # ============ 初始化各个模块实例 ============
        self.recorder = Recorder()
        self.llm = LLM()
        self.stt = SoundToText()
        self.tts = TextToSound()
        self.player = AudioPlayer()
        self.analyzer = ImageAnalyzer()

        # 当前选中的图片（numpy 格式）
        self.current_image = None
        self.current_image_path = ""
        self._display_image_buf = None  # 保留 QImage 引用防止 GC

        # 工作线程列表（防止被垃圾回收）
        self._active_threads = []

        # 流水线是否正在运行（运行期间禁用各按钮）
        self._pipeline_busy = False

        # 播放完成信号连接
        self.sig_playback_done.connect(self._on_playback_done)

        # ============ 初始化界面 ============
        self.init_ui()

    # -------------------- 界面布局（响应式） --------------------

    def init_ui(self):
        """使用布局管理器搭建界面，窗口拉伸时自动跟随缩放"""
        self.setObjectName("MainWindow")
        self.setWindowTitle("多模态智能语音助手")
        self.resize(1100, 760)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(GLOBAL_QSS)
        self.setFont(QFont("Microsoft YaHei", 11))

        # ============ 顶部横幅 ============
        banner = QFrame(self)
        banner.setObjectName("banner")
        banner.setFixedHeight(64)

        title = QLabel("多模态智能语音助手", banner)
        title.setObjectName("bannerTitle")
        title.setAlignment(Qt.AlignCenter)
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        banner_layout.addWidget(title)

        # ============ 整体垂直布局 ============
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(banner)

        # 主体：左右两栏
        body = QHBoxLayout()
        body.setContentsMargins(20, 16, 20, 20)
        body.setSpacing(20)
        outer.addLayout(body, 1)

        # ============ 左栏 ============
        left = QVBoxLayout()
        left.setSpacing(16)
        body.addLayout(left, 5)

        # -------- 左栏：对话记录卡片 --------
        card_chat = QFrame(self)
        card_chat.setObjectName("card")

        chat_v = QVBoxLayout(card_chat)
        chat_v.setContentsMargins(18, 14, 18, 18)
        chat_v.setSpacing(10)

        lbl_chat = QLabel("对话记录", card_chat)
        lbl_chat.setObjectName("cardTitle")
        chat_v.addWidget(lbl_chat)

        self.text_chat = QTextBrowser(card_chat)
        self.text_chat.setMinimumHeight(220)
        self.text_chat.setOpenExternalLinks(False)
        self.text_chat.append(
            "<b style='color:#00A67E;'>AI：</b>你好！我是多模态语音助手，"
            "可以和我语音聊天、识别图片，有什么想说的吗？"
        )
        chat_v.addWidget(self.text_chat, 1)

        left.addWidget(card_chat, 5)

        # -------- 左栏：语音对话卡片 --------
        card_voice = QFrame(self)
        card_voice.setObjectName("card")

        voice_v = QVBoxLayout(card_voice)
        voice_v.setContentsMargins(18, 14, 18, 18)
        voice_v.setSpacing(10)

        lbl_voice = QLabel("语音对话", card_voice)
        lbl_voice.setObjectName("cardTitle")
        voice_v.addWidget(lbl_voice)

        voice_h = QHBoxLayout()
        voice_h.setSpacing(20)

        self.btn_record = QPushButton("开始录音", card_voice)
        self.btn_record.setMinimumSize(190, 54)
        self.btn_record.setStyleSheet(RECORD_BTN_IDLE)
        self.btn_record.clicked.connect(self.on_record_clicked)
        voice_h.addWidget(self.btn_record)

        voice_right = QVBoxLayout()
        voice_right.setSpacing(6)
        self.lbl_status = QLabel("状态：空闲", card_voice)
        self._set_status_style(self.STATE_IDLE)
        voice_right.addWidget(self.lbl_status)

        hint_voice = QLabel("点击开始录音 → 说话 → 再次点击结束，AI 将回复", card_voice)
        hint_voice.setObjectName("hintText")
        voice_right.addWidget(hint_voice)

        voice_h.addLayout(voice_right, 1)
        voice_v.addLayout(voice_h)

        left.addWidget(card_voice)

        # -------- 左栏：文字对话卡片 --------
        card_text = QFrame(self)
        card_text.setObjectName("card")

        text_v = QVBoxLayout(card_text)
        text_v.setContentsMargins(18, 14, 18, 18)
        text_v.setSpacing(10)

        lbl_text = QLabel("文字对话", card_text)
        lbl_text.setObjectName("cardTitle")
        text_v.addWidget(lbl_text)

        text_h = QHBoxLayout()
        text_h.setSpacing(12)

        self.input_text = QLineEdit(card_text)
        self.input_text.setMinimumHeight(44)
        self.input_text.setPlaceholderText("输入你的问题...（回车发送）")
        self.input_text.returnPressed.connect(self.on_text_send)
        text_h.addWidget(self.input_text, 1)

        self.btn_send_text = QPushButton("发送", card_text)
        self.btn_send_text.setObjectName("sendBtn")
        self.btn_send_text.setMinimumSize(120, 44)
        self.btn_send_text.clicked.connect(self.on_text_send)
        text_h.addWidget(self.btn_send_text)

        text_v.addLayout(text_h)
        left.addWidget(card_text)

        # ============ 右栏 ============
        right = QVBoxLayout()
        right.setSpacing(16)
        body.addLayout(right, 4)

        # -------- 右栏：视觉多模态卡片 --------
        card_vision = QFrame(self)
        card_vision.setObjectName("card")

        vision_v = QVBoxLayout(card_vision)
        vision_v.setContentsMargins(18, 14, 18, 18)
        vision_v.setSpacing(10)

        lbl_vision = QLabel("视觉多模态", card_vision)
        lbl_vision.setObjectName("cardTitle")
        vision_v.addWidget(lbl_vision)

        self.lbl_image = QLabel("（此处显示图片）", card_vision)
        self.lbl_image.setMinimumSize(280, 180)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet(
            "border: 2px dashed #C9D2E0; background-color: #F7F9FC; color: #9AA3B2;"
            "border-radius: 10px; font-size: 14px;"
        )
        vision_v.addWidget(self.lbl_image, 1)

        img_btn_h = QHBoxLayout()
        img_btn_h.setSpacing(12)
        self.btn_camera = QPushButton("拍照", card_vision)
        self.btn_camera.setMinimumHeight(42)
        self.btn_camera.clicked.connect(self.on_capture_clicked)
        img_btn_h.addWidget(self.btn_camera, 1)

        self.btn_pick_image = QPushButton("选择图片", card_vision)
        self.btn_pick_image.setMinimumHeight(42)
        self.btn_pick_image.clicked.connect(self.on_pick_image_clicked)
        img_btn_h.addWidget(self.btn_pick_image, 1)
        vision_v.addLayout(img_btn_h)

        img_q_h = QHBoxLayout()
        img_q_h.setSpacing(12)
        self.input_image_q = QLineEdit(card_vision)
        self.input_image_q.setMinimumHeight(40)
        self.input_image_q.setPlaceholderText("对图片的问题（可选）")
        img_q_h.addWidget(self.input_image_q, 1)

        self.btn_analyze = QPushButton("识别图片", card_vision)
        self.btn_analyze.setObjectName("sendBtn")
        self.btn_analyze.setMinimumSize(110, 40)
        self.btn_analyze.clicked.connect(self.on_analyze_image_clicked)
        img_q_h.addWidget(self.btn_analyze)
        vision_v.addLayout(img_q_h)

        right.addWidget(card_vision, 5)

        # -------- 右栏：设置卡片 --------
        card_settings = QFrame(self)
        card_settings.setObjectName("card")

        set_v = QVBoxLayout(card_settings)
        set_v.setContentsMargins(18, 14, 18, 18)
        set_v.setSpacing(10)

        lbl_settings = QLabel("设置", card_settings)
        lbl_settings.setObjectName("cardTitle")
        set_v.addWidget(lbl_settings)

        def make_label(text, parent):
            lab = QLabel(text, parent)
            lab.setFixedWidth(84)
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lab

        row_api = QHBoxLayout()
        row_api.setSpacing(10)
        row_api.addWidget(make_label("API Key：", card_settings))
        self.input_api = QLineEdit(card_settings)
        self.input_api.setMinimumHeight(38)
        self.input_api.setPlaceholderText("硅基流动 API Key")
        self.input_api.setEchoMode(QLineEdit.Password)
        if API_KEY:
            self.input_api.setText(API_KEY)
        row_api.addWidget(self.input_api, 1)
        set_v.addLayout(row_api)

        row_tts = QHBoxLayout()
        row_tts.setSpacing(10)
        row_tts.addWidget(make_label("TTS 服务：", card_settings))
        self.combo_tts = QComboBox(card_settings)
        self.combo_tts.setMinimumHeight(38)
        self.combo_tts.addItem("本地 edge-tts", TTS_LOCAL_SERVICE)
        self.combo_tts.addItem("云端 CosyVoice", TTS_SILICONFLOW)
        idx = self.combo_tts.findData(DEFAULT_TTS_PROVIDER)
        if idx >= 0:
            self.combo_tts.setCurrentIndex(idx)
        self.combo_tts.currentIndexChanged.connect(self.on_tts_provider_changed)
        row_tts.addWidget(self.combo_tts, 1)
        set_v.addLayout(row_tts)

        row_voice = QHBoxLayout()
        row_voice.setSpacing(10)
        row_voice.addWidget(make_label("音色：", card_settings))
        self.combo_voice = QComboBox(card_settings)
        self.combo_voice.setMinimumHeight(38)
        for voice_id, desc in LOCAL_VOICE_OPTIONS:
            self.combo_voice.addItem(desc, voice_id)
        idx = self.combo_voice.findData(DEFAULT_LOCAL_VOICE)
        if idx >= 0:
            self.combo_voice.setCurrentIndex(idx)
        row_voice.addWidget(self.combo_voice, 1)
        set_v.addLayout(row_voice)

        row_search = QHBoxLayout()
        row_search.setSpacing(10)
        row_search.addWidget(make_label("联网搜索：", card_settings))
        self.check_search = QCheckBox("回答前先联网检索", card_settings)
        self.check_search.setChecked(True)
        self.check_search.setFont(QFont("Microsoft YaHei", 12))
        row_search.addWidget(self.check_search, 1)
        set_v.addLayout(row_search)

        self.btn_quit = QPushButton("退出程序", card_settings)
        self.btn_quit.setObjectName("quitBtn")
        self.btn_quit.setMinimumHeight(42)
        self.btn_quit.clicked.connect(self.close)
        set_v.addWidget(self.btn_quit)

        right.addWidget(card_settings)

        # ============ 默认状态 ============
        self._update_state(self.STATE_IDLE)

    # -------------------- 辅助方法 --------------------

    def _set_status_style(self, state_text):
        """根据状态设置状态栏颜色"""
        if state_text == self.STATE_IDLE:
            color = "#6B7280"
            bold = ""
        elif state_text == self.STATE_RECORDING:
            color = "#DC2626"
            bold = "font-weight:bold;"
        elif state_text == self.STATE_PLAYING:
            color = "#00A67E"
            bold = "font-weight:bold;"
        elif state_text == "图片分析中":
            color = "#7C3AED"
            bold = "font-weight:bold;"
        else:
            color = "#EA580C"
            bold = "font-weight:bold;"
        self.lbl_status.setStyleSheet(
            f"color:{color}; {bold} font-size:15px;"
        )

    def _update_state(self, state_text):
        """更新状态栏文字与颜色"""
        self.lbl_status.setText(f"状态：{state_text}")
        self._set_status_style(state_text)

    def _get_api_key(self):
        key = self.input_api.text().strip()
        return key if key else API_KEY

    def _update_modules_api_key(self):
        key = self._get_api_key()
        self.stt.api_key = key
        self.llm.api_key = key
        self.tts.api_key = key
        self.analyzer.api_key = key

    def _get_tts_config(self):
        provider = self.combo_tts.currentData()
        if provider == TTS_LOCAL_SERVICE:
            voice = self.combo_voice.currentData()
        else:
            voice = None  # 云端用默认音色
        return provider, voice

    def _check_api_key(self):
        if not self._get_api_key():
            QMessageBox.warning(
                self, "提示",
                "请先在右侧【设置】区填入硅基流动 API Key\n"
                "注册地址: https://siliconflow.cn"
            )
            return False
        return True

    def _lock_buttons(self):
        self._pipeline_busy = True
        self.btn_record.setEnabled(False)
        self.btn_send_text.setEnabled(False)
        self.input_text.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_pick_image.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.input_image_q.setEnabled(False)
        self.check_search.setEnabled(False)

    def _unlock_buttons(self):
        self._pipeline_busy = False
        self.btn_record.setEnabled(not self.recorder.is_recording)
        self.btn_send_text.setEnabled(True)
        self.input_text.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_pick_image.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.input_image_q.setEnabled(True)
        self.check_search.setEnabled(True)

    def _run_thread(self, thread):
        """安全启动 QThread：加入列表防止 GC，并用内置 finished 信号自动清理"""
        self._active_threads.append(thread)

        def on_thread_finished():
            if thread in self._active_threads:
                self._active_threads.remove(thread)

        thread.finished.connect(on_thread_finished)
        thread.start()

    def closeEvent(self, event):
        """退出时停止播放，释放资源"""
        try:
            self.player.stop()
        except Exception:
            pass
        event.accept()

    # -------------------- 信号槽：语音对话 --------------------

    def on_record_clicked(self):
        """录音按钮点击：切换开始/停止状态"""
        if self._pipeline_busy:
            QMessageBox.information(self, "提示", "正在处理上一轮请求，请稍等...")
            return

        if self.recorder.is_recording:
            self._stop_recording()
        else:
            # 录音已超时自动停止但按钮还显示"结束录音"的情况
            if self.btn_record.text() == "结束录音":
                QMessageBox.information(
                    self, "提示", "录音已超时自动停止，未采集到语音，请重试"
                )
                self.btn_record.setText("开始录音")
                self.btn_record.setStyleSheet(RECORD_BTN_IDLE)
                self._update_state(self.STATE_IDLE)
                return
            self._start_recording()

    def _start_recording(self):
        ok, err = self.recorder.start()
        if not ok:
            QMessageBox.critical(self, "错误", err)
            return

        self.btn_record.setText("结束录音")
        self.btn_record.setStyleSheet(RECORD_BTN_RECORDING)
        self._update_state(self.STATE_RECORDING)

    def _stop_recording(self):
        audio_buf, err = self.recorder.stop()

        self.btn_record.setText("开始录音")
        self.btn_record.setStyleSheet(RECORD_BTN_IDLE)

        if err or audio_buf is None:
            if err:
                if err == "当前没有在录音":
                    QMessageBox.information(self, "提示", "录音已超时自动停止，请重试")
                else:
                    QMessageBox.warning(self, "提示", err)
            self._update_state(self.STATE_IDLE)
            return

        # 执行完整链路：ASR → LLM → TTS → 播放
        self._run_voice_pipeline(audio_buf)

    def _run_voice_pipeline(self, audio_buffer):
        """完整语音对话链路：ASR → LLM → TTS → 播放（各步骤串行）"""
        if not self._check_api_key():
            return
        self._update_modules_api_key()

        self._lock_buttons()

        # -------- 第一步：ASR 语音识别 --------
        self._update_state(self.STATE_ASR)
        self._append_chat("system", "正在识别语音...")

        def step1_asr_done(text, err):
            if err:
                self._append_chat("ai", f"语音识别失败：{err}")
                self._update_state(self.STATE_IDLE)
                self._unlock_buttons()
                return
            if not text:
                self._append_chat("ai", "没有听清你说的话，请再试一次")
                self._update_state(self.STATE_IDLE)
                self._unlock_buttons()
                return

            # 识别成功，显示用户消息
            self._append_chat("user", text)

            # 用户说"退出"则结束本轮
            if text.strip() in ("退出", "结束", "再见", "拜拜"):
                self._append_chat("ai", "好的，再见！")
                self._update_state(self.STATE_IDLE)
                self._unlock_buttons()
                return

            # -------- 第二步：联网搜索（可选）→ LLM 流式回复（简洁模式） --------
            self._run_search_then_llm(
                text, lambda reply: self._run_tts(reply), concise_hint=100
            )

        thread = ASRThread(self.stt, audio_buffer)
        thread.done.connect(step1_asr_done)
        self._run_thread(thread)

    def _run_tts(self, text):
        """执行 TTS 合成 + 播放（超长回复只播报前段）"""
        provider, voice = self._get_tts_config()
        self.tts = TextToSound(provider=provider, voice=voice,
                               api_key=self._get_api_key())

        # 语音播报截断：完整回复仍显示在对话区
        text_for_tts = text
        if len(text_for_tts) > TTS_MAX_TEXT_CHARS:
            text_for_tts = self._truncate_sentence(text_for_tts, TTS_MAX_TEXT_CHARS)
            self._append_chat(
                "system",
                f"语音回复过长，仅播报前 {len(text_for_tts)} 字，完整内容见上方对话记录",
            )

        self._update_state(self.STATE_TTS)

        def step_tts_done(audio_bytes, err):
            if err or not audio_bytes:
                reason = err if err else "语音合成失败"
                self._append_chat("ai", f"\n\n（{reason}，仅显示文字回复）")
                self._update_state(self.STATE_IDLE)
                self._unlock_buttons()
                return
            # -------- 第四步：播放语音 --------
            self._update_state(self.STATE_PLAYING)
            ok = self.player.play(audio_bytes, on_finished=self.sig_playback_done.emit)
            if not ok:
                # 播放失败（如临时文件被占用）也要解除锁定，避免卡死
                self._append_chat("ai", "\n\n（语音播放失败，仅显示文字回复）")
                self._update_state(self.STATE_IDLE)
                self._unlock_buttons()

        thread = TTSThread(self.tts, text_for_tts)
        thread.done.connect(step_tts_done)
        self._run_thread(thread)

    @staticmethod
    def _truncate_sentence(text, max_chars):
        """在句子边界截断文本，避免播报中途断句"""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        idx = max(cut.rfind("。"), cut.rfind("！"), cut.rfind("？"),
                  cut.rfind("."), cut.rfind("! "), cut.rfind("? "))
        if idx >= max_chars * 0.5:
            return cut[:idx + 1]
        return cut

    # -------------------- 联网搜索 + LLM 回复（文字/语音共用） --------------------

    def _run_search_then_llm(self, text, on_reply_done, concise_hint=None):
        """可选联网搜索 → LLM 流式回复
        :param text: 用户问题
        :param on_reply_done: 成功拿到完整回复后调用，参数为回复文本
        :param concise_hint: 简洁回答字数上限（语音场景用，缩短播报时长）
        """
        if self.check_search.isChecked():
            self._update_state("联网搜索中")
            self._append_chat("system", "正在联网搜索相关资料...")

            def on_search_done(results, err):
                if err:
                    self._append_chat("system", f"联网搜索失败：{err}，将直接回答")
                    context = ""
                elif results:
                    context = self._format_search_results(results)
                    self._append_chat(
                        "system", f"已获取 {len(results)} 条搜索结果"
                    )
                else:
                    context = ""
                    self._append_chat("system", "未搜到相关结果，将直接回答")
                self._start_llm(text, context, on_reply_done, concise_hint)

            thread = SearchThread(text)
            thread.done.connect(on_search_done)
            self._run_thread(thread)
        else:
            self._start_llm(text, "", on_reply_done, concise_hint)

    def _start_llm(self, text, context, on_reply_done, concise_hint=None):
        """启动 LLM 流式回复（带可选搜索上下文与简洁提示）"""
        self._update_state(self.STATE_LLM)
        self._append_chat("ai", "")

        def on_token(token):
            self.text_chat.moveCursor(self.text_chat.textCursor().End)
            self.text_chat.insertPlainText(token)

        def on_done(reply, err):
            if err:
                self._append_chat("ai", f"\n\n大模型调用失败：{err}")
                self._update_state(self.STATE_IDLE)
                self._unlock_buttons()
                return
            on_reply_done(reply)

        thread = LLMStreamThread(self.llm, text, context, concise_hint)
        thread.token_received.connect(on_token)
        thread.done.connect(on_done)
        self._run_thread(thread)

    def _format_search_results(self, results):
        """把搜索结果格式化成给大模型的参考文本"""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"{i}. {title}　{url}\n　　{snippet}")
        return "\n".join(lines)

    def _on_playback_done(self):
        """语音播放完成（主线程）"""
        self._update_state(self.STATE_IDLE)
        self._unlock_buttons()

    # -------------------- 信号槽：文字对话 --------------------

    def on_text_send(self):
        if self._pipeline_busy:
            QMessageBox.information(self, "提示", "正在处理上一轮请求，请稍等...")
            return

        text = self.input_text.text().strip()
        if not text:
            return

        if not self._check_api_key():
            return
        self._update_modules_api_key()

        self.input_text.clear()
        self._lock_buttons()

        self._append_chat("user", text)

        # 联网搜索（可选）→ LLM 流式回复
        self._run_search_then_llm(
            text,
            lambda reply: (self._update_state(self.STATE_IDLE),
                           self._unlock_buttons()),
        )

    # -------------------- 信号槽：视觉多模态 --------------------

    def on_capture_clicked(self):
        image, err = self.analyzer.capture_from_camera()
        if err:
            QMessageBox.warning(self, "拍照失败", err)
            return
        self.current_image = image
        self.current_image_path = ""
        self._display_image(image)

    def on_pick_image_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if not path:
            return

        image, err = self.analyzer.load_from_file(path)
        if err:
            QMessageBox.warning(self, "读取失败", err)
            return
        self.current_image = image
        self.current_image_path = path
        self._display_image(image)

    def _display_image(self, image_np):
        """把 numpy 图片显示到 QLabel 上（跟随显示区大小缩放）"""
        try:
            rgb_image = np.ascontiguousarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            h, w, ch = rgb_image.shape

            self._display_image_buf = rgb_image.tobytes()

            q_img = QImage(self._display_image_buf, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            # 跟随当前显示区大小缩放
            target = self.lbl_image.size()
            scaled = pixmap.scaled(
                max(target.width(), 1), max(target.height(), 1),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_image.setPixmap(scaled)
            self.lbl_image.setStyleSheet(
                "border: 2px solid #00A67E; border-radius: 10px; background-color: #F7F9FC;"
            )
        except Exception as e:
            self.lbl_image.setText("图片显示失败")
            print(f"图片显示异常: {e}")

    def on_analyze_image_clicked(self):
        if self._pipeline_busy:
            QMessageBox.information(self, "提示", "正在处理上一轮请求，请稍等...")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先拍照或选择一张图片")
            return

        if not self._check_api_key():
            return
        self._update_modules_api_key()

        question = self.input_image_q.text().strip() or "请描述下这张图片"

        self._append_chat("user", f"[图片问题] {question}")
        self._append_chat("ai", "")

        self._update_state("图片分析中")
        self._lock_buttons()

        def on_token(token):
            self.text_chat.moveCursor(self.text_chat.textCursor().End)
            self.text_chat.insertPlainText(token)

        def on_done(reply, err):
            if err:
                self._append_chat("ai", f"\n\n图片分析失败：{err}")
            self._update_state(self.STATE_IDLE)
            self._unlock_buttons()

        thread = VisionThread(self.analyzer, self.current_image.copy(), question)
        thread.token_received.connect(on_token)
        thread.done.connect(on_done)
        self._run_thread(thread)

    # -------------------- 信号槽：其他 --------------------

    def on_tts_provider_changed(self, index):
        """TTS 提供商切换时更新音色下拉框可用状态"""
        provider = self.combo_tts.currentData()
        self.combo_voice.setEnabled(provider == TTS_LOCAL_SERVICE)

    # -------------------- 对话框追加 --------------------

    def _append_chat(self, role, text):
        """
        向对话区追加一条消息
        :param role: 'user' / 'ai' / 'system'
        :param text: 消息内容（为空则只加一个占位，用于后续流式追加）
        """
        if role == "user":
            self.text_chat.append(f"<b style='color:#2B7DE9;'>用户：</b>{text}")
        elif role == "ai":
            if text:
                self.text_chat.append(f"<b style='color:#00A67E;'>AI：</b>{text}")
            else:
                self.text_chat.append("<b style='color:#00A67E;'>AI：</b>")
        elif role == "system":
            self.text_chat.append(f"<i style='color:#9AA3B2;'>{text}</i>")

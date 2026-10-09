# -*- coding: utf-8 -*-
"""
集中配置文件
所有 API Key、模型名、默认参数都放在这里，方便统一管理

【密钥策略】本文件会进 git，所以不要在这里写密钥。
密钥放在同目录的 config.local.json（已在 .gitignore 中），或环境变量：
    优先级：环境变量 > config.local.json > 本文件默认值
    SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL

【路径策略】所有路径都基于"本文件所在目录"算绝对路径。
不要改回相对路径——相对路径依赖启动时的工作目录，
用 IDE 运行和用命令行运行会静默读写到不同目录（踩过这个坑）。
"""

import json
import os

import pyaudio  # 需要 pyaudio 常量（paInt16 等），不要用数字硬编码

# ============ 基础路径 ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.local.json")


def _load_local_config():
    """
    读取 config.local.json（不存在或损坏都不影响启动）
    用 utf-8-sig：Windows 上很容易写出带 BOM 的 UTF-8，用普通 utf-8 读会抛异常，
    导致密钥"静默丢失"（踩过这个坑）。
    """
    try:
        with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        import sys
        sys.stderr.write("[config] config.local.json 读取失败（密钥可能未生效）: %s\n" % e)
        return {}


_LOCAL = _load_local_config()


def _pick(env_name, local_key, default=""):
    """按 环境变量 > config.local.json > 默认值 的顺序取值"""
    value = os.environ.get(env_name)
    if value and value.strip():
        return value.strip()
    value = _LOCAL.get(local_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


# ============ 硅基流动 SiliconFlow API 配置 ============
# 把密钥写进 config.local.json（{"API_KEY": "sk-..."}）或设环境变量 SILICONFLOW_API_KEY
# 注册地址: https://siliconflow.cn
API_KEY = _pick("SILICONFLOW_API_KEY", "API_KEY", "")

# 统一的 Base URL（所有 AI 服务都走硅基流动）
BASE_URL = _pick("SILICONFLOW_BASE_URL", "BASE_URL", "https://api.siliconflow.cn/v1")

# ============ 模型名称 ============
# 语音识别（ASR）模型：SenseVoiceSmall
# 实测：高基频(+150Hz)/快语速(+25%)/短片段均能全对；
#       老年声学逼近(-40Hz/-25%语速/-50%音量)平均相似度 0.979，唯一错误是"药"→"要"（同音字）
STT_MODEL = "FunAudioLLM/SenseVoiceSmall"

# 大语言模型（LLM）：Qwen2.5-7B-Instruct
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 视觉多模态模型（VL）：Qwen3-VL-8B-Instruct
# 注：课程笔记里的 Pro/Qwen/Qwen2.5-VL-7B-Instruct 在硅基流动已下线（返回 Model disabled），
#     换成当前可用的 Qwen3-VL 系列。想要更强效果可改为 "Qwen/Qwen3-VL-32B-Instruct"。
VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# 云端语音合成模型（CosyVoice2）
TTS_SILICONFLOW_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
# CosyVoice2 的音色参数
TTS_SILICONFLOW_VOICE = "FunAudioLLM/CosyVoice2-0.5B:alex"

# ============ 大模型参数 ============
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 300          # 语音场景回答要短，但 150 容易截断，取 300
LLM_SYSTEM_PROMPT = "你是一个助手"

# 是否让模型"思考"（推理模型专用）
# 实测 Qwen3.5 系列思考开启时首字延迟 37 秒，关闭后 0.79 秒（差 47 倍）。
# 语音场景必须关掉。若模型不支持该参数，core/llm.py 会自动去掉它重试。
LLM_ENABLE_THINKING = False

# 单次请求超时与重试。必须设置：不设时 SDK 默认 600 秒，
# 一次网络卡住会让整个语音链路无限等待（实测卡过 9 分钟无响应）。
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 1

# ============ 语音识别（ASR）的延迟兜底 ============
# 实测 ASR 稳态延迟 0.2~0.4 秒，但抖动可达 60 秒以上（云端服务与网络所致），
# 因此设计三层：启动预热 → 超时重试 → 超时后降级提示用户重说
ASR_TIMEOUT_SECONDS = 20      # 单次识别超时
ASR_MAX_RETRIES = 1           # 超时后重试次数
ASR_WARMUP_ON_START = True    # 启动时发一次极短音频预热，避免首次冷启动 2~3 秒

# ============ 录音参数（关键！） ============
# RECORD_FORMAT 必须写成 pyaudio.paInt16（它的值是 8）。
# ⚠️ 千万别写成数字 1：1 是 paFloat32，那样录到的会是 32 位浮点数据，
#    再按 16 位整数封装成 WAV 就全是坏数据，语音识别自然"听不到"。
RECORD_FORMAT = pyaudio.paInt16

# 声道数：笔记本麦克风普遍只支持双声道，设 2 最稳。
# 录音完成后 Recorder 会自动把双声道降混为单声道再交给语音识别。
RECORD_CHANNELS = 2

RECORD_RATE = 16000       # 采样率 16kHz（SenseVoiceSmall 推荐）
RECORD_CHUNK = 1024       # 每次读取的帧数
RECORD_MAX_SECONDS = 30   # 最长录音时间（秒），防止用户忘记停止

# 麦克风设备索引
# None = 使用系统默认麦克风
# 运行 find_mic.py 找到能用的索引后填在这里，比如 RECORD_DEVICE = 1
RECORD_DEVICE = None

# ============ 连续对话：VAD 自动断句 ============
# 目标是"点一次开始，之后免按键"：靠语音活动检测自动切出一句话
VAD_ENABLED = True
VAD_MODE = 2                  # webrtcvad 激进度 0~3，越大越容易把噪声判成语音；老年用户取 2
VAD_FRAME_MS = 30             # webrtcvad 只接受 10/20/30 毫秒帧，30 最稳
SILENCE_END_MS = 800          # 说完后静音多久算一句话结束（老人说话有停顿，不能太短）
MIN_SPEECH_MS = 200           # 少于这么久认为是咳嗽/碰撞等噪声，丢弃
MAX_UTTERANCE_SECONDS = 20    # 单句上限，超过强制切分（避免用户忘停）
VAD_PADDING_MS = 200          # 首尾各留一点，防止把字头字尾切掉

# ============ 播报中打断（barge-in） ============
# 最难的一环：播报时麦克风会听见扬声器的声音，必须区分"自己说话"和"用户插话"
BARGE_IN_ENABLED = True
# 阈值由实测校准：稳态回声段的"能量/本底"比值约 1~4，
# 用户说话时可达 30~50，取 3.0 能干净分开（取 1.8 会一直被回声误触发）
BARGE_IN_ENERGY_RATIO = 3.0
BARGE_IN_MIN_MS = 300         # 持续这么久才判定为打断（防误触发）
BARGE_IN_GRACE_MS = 300       # 开始播报后的保护期，避免起播瞬间误判

# ============ 回声消除（AEC） ============
# 用 WASAPI 回环采集"系统正在播放的声音"作为参考信号，与麦克风信号做自适应抵消。
# 没有它，播报时麦克风会听见自己，连续对话与打断都无法工作。
AEC_ENABLED = True
# 滤波器长度：只需覆盖"扬声器→麦克风"的直达声与早期反射（典型 10~50ms），
# 取 100ms 足够。注意 CPU 开销随长度平方增长（时域卷积），别盲目加大。
AEC_FILTER_MS = 100
AEC_STEP_SIZE = 0.15          # NLMS 步长：太大不稳定，太小收敛慢
AEC_BLOCK_SAMPLES = 160       # 每次处理的块长（10ms），影响自适应速度与 CPU
AEC_REF_DEVICE_INDEX = None   # None = 自动找第一个回环设备

# ============ 唤醒词（可选，离线） ============
WAKE_WORD_ENABLED = False     # 默认关闭：先用"点一次按钮"的连续对话，稳定后再开
WAKE_WORD_MODEL = "hey_jarvis"   # openWakeWord 内置模型名；自定义唤醒词需另训

# ============ 本地 edge-tts 音色列表 ============
# 来自课程笔记的中文音色
LOCAL_VOICE_OPTIONS = [
    ("zh-CN-YunxiNeural", "男 · 阳光（默认）"),
    ("zh-CN-XiaoxiaoNeural", "女 · 柔和"),
    ("zh-CN-XiaoyiNeural", "女 · 活泼"),
    ("zh-CN-YunjianNeural", "男 · 激情"),
    ("zh-CN-YunyangNeural", "男 · 播音"),
]
DEFAULT_LOCAL_VOICE = "zh-CN-YunxiNeural"

# 适老化：默认放慢语速（老人听清比听快重要）
DEFAULT_SPEECH_RATE = "-10%"

# ============ TTS 提供商选项 ============
TTS_LOCAL_SERVICE = "LOCAL_SERVICE"   # 本地 edge-tts（免费）
TTS_SILICONFLOW = "SILICONFLOW"       # 云端 CosyVoice2（需 API Key）
DEFAULT_TTS_PROVIDER = TTS_LOCAL_SERVICE

# 语音合成总超时（秒）：网络卡住时不再无限等待，超时自动降级为文字回复
TTS_TIMEOUT_SECONDS = 60

# 语音播报最长字符数：超过则只播报前段（完整回复仍在对话区显示），
# 避免长回复产生 30~40 秒的语音、界面长时间锁定
TTS_MAX_TEXT_CHARS = 120

# 分句流式播报：模型每产出一句就合成并立即播放，不必等全文。
# 这是"首响延迟"的关键，比等全部生成完再合成快得多。
TTS_STREAM_BY_SENTENCE = True
TTS_SENTENCE_MIN_CHARS = 8    # 太短的句子先攒一攒，避免碎成一个个词

# ============ 适老化界面与大字幕 ============
UI_FONT_SIZE = 16             # 基础字号（普通用户 11~12，这里明显放大）
UI_SUBTITLE_FONT_SIZE = 28    # 字幕字号，给听障/视力下降用户看
UI_HIGH_CONTRAST = True       # 高对比配色
SUBTITLE_MAX_ITEMS = 200      # 字幕历史保留条数
SUBTITLE_ALWAYS_ON_TOP = True # 字幕窗口置顶，方便对方看着读

# ============ 确认式交互（关键信息必须复述） ============
# 实测 ASR 会把"这个药"听成"这个要"，用药/金额/时间这类信息必须二次确认
CONFIRM_ENABLED = True
CONFIRM_KEYWORDS = ("药", "吃几", "剂量", "钱", "多少钱", "金额", "转账",
                    "几点", "时间", "日期", "预约", "密码")

# ============ 手语 / 手势（视觉通道） ============
# 完整手语是"手形+位置+运动+表情"的语法系统，全量识别属研究级难题。
# 本项目定位受限词表：高频沟通手语 + 数字手势 + 控制手势。
HAND_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
HAND_MAX_HANDS = 2
HAND_DETECT_CONFIDENCE = 0.5
GESTURE_SAMPLE_INTERVAL_MS = 100    # 采样间隔，太密费 CPU、太疏漏动作

# 内置常用手语词（对应聋哑用户的高频表达需求）
COMMON_SIGNS = ("喝水", "吃饭", "上厕所", "疼", "吃药", "帮忙",
                "谢谢", "不要", "等一下", "医院", "回家", "打电话")

# 控制手势 -> 动作（免手操作程序本身）
GESTURE_CONTROLS = {
    "open_palm": "开始/停止聆听",
    "fist": "取消当前输入",
    "thumbs_up": "确认发送",
    "thumbs_down": "停止播报",
    "victory": "切换字幕模式并放大字号",
    "wave": "紧急停止（断麦断摄像头）",
}

# ============ 隐私开关 ============
# 摄像头与麦克风各有独立开关；一键"紧急停止"会同时断开
PRIVACY_DEFAULT_CAMERA_ON = False
PRIVACY_DEFAULT_MIC_ON = True

# ============ 临时文件路径 ============
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
TEMP_REPLY_FILE = os.path.join(TEMP_DIR, "reply.mp3")

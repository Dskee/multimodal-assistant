# 多模态智能语音助手

一个基于 **PyQt5** 的 Windows 桌面应用，集成了语音对话、视觉多模态、文字对话三大能力。

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🎤 语音对话 | 点击录音 → 语音识别 → 大模型回复 → 语音合成 → 自动播放（4 种中文音色可选） |
| 📷 视觉多模态 | 支持摄像头拍照或本地图片，用 VL 模型分析，可对图片提问 |
| ⌨️ 文字对话 | 文本框输入，大模型流式回复 |
| 🌐 联网搜索 | 回答前自动联网检索（DuckDuckGo + Bing 并行合并，无需 API Key，10 秒硬预算），可在设置中关闭 |

## 🛠️ 技术栈

- **Python 3.9**（Anaconda base）+ **PyQt5 5.15.11**
- **AI 服务**：硅基流动 SiliconFlow（统一 `base_url = https://api.siliconflow.cn/v1`）
  - 语音识别：`FunAudioLLM/SenseVoiceSmall`
  - 大模型：`Qwen/Qwen2.5-7B-Instruct`
  - 视觉多模态：`Qwen/Qwen3-VL-8B-Instruct`
    （注：课程笔记中的 `Pro/Qwen/Qwen2.5-VL-7B-Instruct` 在硅基流动已下线，会返回 "Model disabled"；换成 Qwen3-VL 系列即可，想更强可改 `Qwen/Qwen3-VL-32B-Instruct`）
- **语音合成**：本地 `edge-tts` + 云端 `CosyVoice2`（可切换）
- **录音**：`pyaudio`（输出 16bit 单声道 16kHz WAV，**无需 ffmpeg**）
- **播放**：`pygame.mixer`
- **视觉**：`opencv-python`

## 🚀 快速开始（推荐）

在 PowerShell 中进入项目目录，直接用 python 运行（`python` 为 Anaconda base 环境，依赖已齐全）：

```powershell
cd D:\VibeCoding\multimodal_assistant
python main.py
```

> 说明：
> - 项目 `libs/` 目录内置了兼容版 `typing_extensions`（base 里 openai 2.x 需要 >=4.5，而全局 site-packages 是 4.3.0 且被系统保护无法覆盖）。各模块在导入 openai 前会自动把 `libs` 加入搜索路径，无需额外安装。
> - 也可以双击 `run.bat` 启动（自动使用 `venv\Scripts\python.exe`）。`venv` 是备用的独立虚拟环境，可保留也可删除。
> - **响应速度**：语音链路各环节都有保护——联网搜索最多 10 秒（超时直接回答）、TTS 合成最多 60 秒（超时自动降级为文字）、语音播报超 100 字自动截断（完整回复仍在对话区），不会无限等待卡死。

## 🔑 获取 API Key

1. 打开 [https://siliconflow.cn](https://siliconflow.cn) 注册账号
2. 登录后进入 **API 密钥管理**，创建一个新密钥
3. 两种填入方式：
   - **方式一**：直接编辑 `config.py`，把 `API_KEY` 改成你的密钥
   - **方式二**：运行程序后，在右侧【设置】区的 API Key 输入框中粘贴

## 📂 项目结构

```
multimodal_assistant/
├── main.py              # 程序入口
├── config.py            # 集中配置（API Key、模型名、参数）
├── run.bat              # 一键启动脚本（推荐）
├── core/
│   ├── recorder.py      # 录音模块（pyaudio，输出 16bit 单声道 WAV）
│   ├── stt.py           # 语音识别 ASR（SenseVoiceSmall）
│   ├── llm.py           # 大模型 LLM（Qwen2.5-7B，流式）
│   ├── tts.py           # 语音合成 TTS（edge-tts / CosyVoice2）
│   ├── player.py        # 音频播放（pygame.mixer）
│   └── vision.py        # 视觉多模态（Qwen3-VL-8B）
├── ui/
│   └── main_window.py   # 主窗口 + QThread 工作线程
├── venv/                # 项目虚拟环境
├── temp/                # 临时文件目录（TTS 生成的音频）
└── requirements.txt     # 依赖列表
```

## 🎯 使用流程

### 语音对话（核心闭环）
1. 点击左侧 **「开始录音」** 按钮，按钮变红
2. 对着麦克风说话（最长 30 秒自动停止）
3. 点击 **「结束录音」**，按钮文字切回"开始录音"
4. 状态依次显示：**语音识别中 → 大模型回复中 → 语音合成 → 语音播报**
5. AI 的语音回答会自动通过扬声器播放，文字同时显示在对话区

### 文字对话
1. 在「文字对话」输入框中输入问题
2. 按 **回车** 或点击 **「发送」** 按钮
3. 大模型会 **流式** 回复

### 图片分析
1. 点击 **「拍照」**（调摄像头）或 **「选择图片」**（本地文件）
2. （可选）在「对图片的问题」框中输入针对性问题，比如"找到图中的公交车"
3. 点击 **「识别图片」** 按钮
4. 多模态模型会分析图片并回复

### 设置区
- **API Key**：密码输入模式，填入硅基流动密钥
- **TTS 服务**：`本地 edge-tts`（免费，推荐）/ `云端 CosyVoice`（需 API Key）
- **音色**：仅本地模式可选，4 种男声女声

## ⚠️ 常见问题

| 问题 | 解决方案 |
|------|----------|
| 语音识别返回"没有听清" | 已修复：旧代码把 `RECORD_FORMAT` 写成 `1`（实为 paFloat32），录出的 WAV 是坏数据。确认 `config.py` 中 `RECORD_FORMAT = pyaudio.paInt16`；另外对着麦克风说话音量要够 |
| 启动报 openai 导入错误 `cannot import name 'override'` | 项目 `libs/` 已内置 typing_extensions 4.16.0 并自动注入路径，直接 `python main.py` 即可；如仍报错请确认 `libs/typing_extensions.py` 存在 |
| 图片识别报 "Model disabled" | 老 VL 模型已下线，`config.py` 已换成 `Qwen/Qwen3-VL-8B-Instruct`；如仍报错，到硅基流动控制台确认模型已开通 |
| API 报错 401 / 403 | API Key 未填或错误，检查右侧设置区 |
| API 报错 429 | 请求频率超限，等几秒再试，或硅基流动账户需充值 |
| 录音报错"麦克风打开失败" | 检查麦克风是否连接、是否被其他程序占用；运行 `find_mic.py` 可逐个测试设备，找到可用索引后填入 `config.py` 的 `RECORD_DEVICE` |

## 🧪 模块单独测试

每个 core 模块都带 `if __name__ == "__main__":` 测试代码，在项目目录下直接运行：

```powershell
# 测试录音（生成 test_record.wav）
python core\recorder.py

# 测试语音识别（需要先有 test_record.wav）
python core\stt.py

# 测试大模型
python core\llm.py

# 测试 TTS + 播放
python core\player.py

# 测试视觉（传图片路径参数，或无参数用摄像头）
python core\vision.py D:\path\to\image.jpg
```

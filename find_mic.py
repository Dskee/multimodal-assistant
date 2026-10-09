# -*- coding: utf-8 -*-
"""
========== 麦克风诊断脚本 ===========
请在你自己的 PowerShell 里运行: python find_mic.py
它会逐个测试你的麦克风，告诉你哪个能识别出人声
"""
import sys, os, time, io, wave, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"))

import pyaudio
from openai import OpenAI

from config import API_KEY, BASE_URL, STT_MODEL

if not API_KEY:
    print('[ERROR] 未找到 API Key：请写进 config.local.json 的 API_KEY，或设环境变量 SILICONFLOW_API_KEY')
    sys.exit(1)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

RATE = 16000
DURATION = 2.5
CHUNK = 1024

print('=' * 65)
print('🎤 麦克风诊断 - 逐个设备测试')
print('=' * 65)
print()

pa = pyaudio.PyAudio()
num_devices = pa.get_device_count()
print(f'共有 {num_devices} 个音频设备')
print()

working_devices = []

for dev_idx in range(num_devices):
    info = pa.get_device_info_by_index(dev_idx)
    name = info['name']
    in_ch = info['maxInputChannels']
    out_ch = info['maxOutputChannels']

    # 跳过没有输入通道的设备
    if in_ch < 1:
        continue

    print(f'--- [{dev_idx}] {name} ({in_ch} in) ---')

    # 尝试打开这个设备
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=min(in_ch, 2),
            rate=RATE,
            input=True,
            input_device_index=dev_idx,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        print(f'    ❌ 打不开: {e}')
        continue

    # 录音
    print(f'    录音中({DURATION}s)...请对着麦克风说"测试语音识别"')
    frames = []
    for _ in range(int(RATE / CHUNK * DURATION)):
        try:
            data = stream.read(CHUNK)
            frames.append(data)
        except Exception as e:
            print(f'    读取出错: {e}')
            break

    stream.stop_stream()
    stream.close()

    if not frames:
        print(f'    ❌ 没录到数据')
        continue

    raw = b''.join(frames)

    # 分析能量
    samples = np.frombuffer(raw, dtype=np.int16)
    nch = min(in_ch, 2)
    if nch > 1:
        samples = samples.reshape(-1, nch)[:, 0]
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(samples)))

    if peak < 10:
        print(f'    ❌ 全静音 (RMS={rms:.0f} Peak={peak:.0f})')
        continue

    # 转 WAV 上传 ASR（双声道先降混为单声道）
    all_samples = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        all_samples = all_samples.reshape(-1, nch).mean(axis=1).astype(np.int16)
    mono_raw = all_samples.tobytes()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
        wf.writeframes(mono_raw)
    buf.seek(0)

    try:
        buf2 = io.BytesIO(buf.getvalue())
        resp = client.audio.transcriptions.create(
            file=('test.wav', buf2, 'audio/wav'),
            model=STT_MODEL,
        )
        text = resp.text.strip()
        if text:
            print(f'    ✅ ASR: "{text}"')
            print(f'       ↓↓↓ 推荐在 config.py 里加一行: RECORD_DEVICE = {dev_idx}')
            working_devices.append((dev_idx, name, text))
        else:
            print(f'    ❌ ASR 返回空 (RMS={rms:.0f})')
    except Exception as e:
        print(f'    ❌ ASR 错误: {e}')

    time.sleep(0.3)

pa.terminate()

print()
print('=' * 65)
if working_devices:
    print(f'✅ 找到 {len(working_devices)} 个能用的麦克风:')
    for idx, name, text in working_devices:
        print(f'   [{idx}] {name} → 识别出: "{text}"')
    print()
    print('请告诉我哪个设备索引能用（比如 "1" 或 "9"），我帮你加到 config.py')
else:
    print('❌ 没有找到能识别出人声的设备')
    print('建议: 检查 Windows 设置 → 声音 → 输入 里的麦克风选择')
print('=' * 65)
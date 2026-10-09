# -*- coding: utf-8 -*-
"""
连续对话引擎：把「音频前端 + 识别 + 大模型 + 分句流式播报」串成一条可用链路

【与旧版语音链路的区别】
  旧版（main_window 里的 _run_voice_pipeline）：
      点按钮 → 录一段 → ASR → LLM（等全文出完）→ TTS（等全部合成）→ 播
      首响延迟 = ASR + LLM 全长 + TTS 全长，很容易 8~15 秒
  新版：
      连续聆听（VAD 自动断句，免按键）→ ASR（带超时与重试）→ LLM 流式
      → **每产出一句就合成一句并立刻播** → 播报中可被打断 → 自动回到聆听

【为什么分句流式是重点】
  它是"首响延迟"的关键：不等全文，第一句一出来就能出声。
  长回复也不必让用户干等十几秒。

【线程模型】
  采集线程（ContinuousListener） → 回调把"一句话"放进任务队列
  工作线程（本类）                → ASR → LLM → 播报，串行处理一个任务
  播报线程（SpeechPlayer）        → 逐句合成 + 播放，可被 interrupt 打断
  三类线程之间只用队列与 Event 通信，不共享可变状态。
"""

import io
import os
import queue
import threading
import time
import wave

from config import (
    RECORD_RATE, TEMP_DIR, TEMP_REPLY_FILE,
    ASR_MAX_RETRIES,
    TTS_LOCAL_SERVICE, DEFAULT_TTS_PROVIDER, DEFAULT_LOCAL_VOICE,
    DEFAULT_SPEECH_RATE, TTS_TIMEOUT_SECONDS, TTS_MAX_TEXT_CHARS,
    TTS_STREAM_BY_SENTENCE, TTS_SENTENCE_MIN_CHARS, TTS_SOFT_CUT_MIN_CHARS,
    TTS_FILLER_ENABLED, TTS_FILLERS,
    CONFIRM_ENABLED, CONFIRM_KEYWORDS, CONFIRM_YES, CONFIRM_NO,
    CONFIRM_TIMEOUT_SECONDS, CONFIRM_MAX_ANSWER_CHARS,
)

# 状态常量（界面据此显示"聆听中/识别中/思考中/播报中"）
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_RECOGNIZING = "recognizing"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"

STATE_LABELS = {
    STATE_IDLE: "空闲",
    STATE_LISTENING: "聆听中",
    STATE_RECOGNIZING: "识别中",
    STATE_THINKING: "思考中",
    STATE_SPEAKING: "播报中",
}


# ======================================================================
# 工具
# ======================================================================

def pcm_to_wav_bytes(pcm, sample_rate=RECORD_RATE):
    """16bit 单声道 PCM → WAV 字节（ASR 需要容器格式，裸 PCM 传上去会识别失败）"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf


def read_wav_to_pcm(path):
    """读一个 WAV 文件成 16bit 单声道 16kHz PCM（测试与"音频文件输入"用）"""
    import wave as _wave
    import numpy as np
    with _wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("只支持 16bit WAV")
    if channels > 1:
        a = np.frombuffer(frames, dtype=np.int16)
        n = (len(a) // channels) * channels
        frames = a[:n].reshape(-1, channels).mean(axis=1).astype(np.int16).tobytes()
    if rate != RECORD_RATE:
        # 简单线性重采样（测试素材用；生产链路的 48k→16k 在 continuous_listener 里）
        import numpy as np
        a = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        idx = np.arange(0, len(a) - 1, float(rate) / RECORD_RATE)
        i0 = idx.astype(int)
        frac = idx - i0
        out = (a[i0] * (1 - frac) + a[i0 + 1] * frac).astype(np.int16)
        frames = out.tobytes()
    return frames


# ======================================================================
# 分句器
# ======================================================================

class SentenceSplitter:
    """
    把流式 token 攒成句子。

    为什么要它：TTS 是按"句"合成最自然，而且第一句一出来就能先播，
    不必等模型把整段话写完（这正是首响延迟的关键）。
    短句会先攒一攒，避免碎成一个个词。
    """

    ENDINGS = "。！？；!?;\n"
    SOFT = "，,、"
    # 硬句末标点的最短句长：只要能构成一句话（≥4 字）就立刻切。
    # 不设这个下限会把"好的。"/"嗯。"这类切出来，TTS 请求碎成一片；
    # 但设太大（比如 8）会把短句和下一句合并，白白拖慢首响（踩过）。
    HARD_MIN_CHARS = 4

    def __init__(self, min_chars=TTS_SENTENCE_MIN_CHARS, max_chars=TTS_MAX_TEXT_CHARS,
                 soft_min_chars=TTS_SOFT_CUT_MIN_CHARS):
        self.min_chars = max(4, int(min_chars))
        self.max_chars = max(self.min_chars + 8, int(max_chars))
        self.soft_min_chars = max(self.HARD_MIN_CHARS, int(soft_min_chars))
        self._buf = ""

    def feed(self, token):
        """喂 token，返回可以播报的句子列表（可能为空）"""
        self._buf += token or ""
        out = []
        while True:
            cut = self._find_cut()
            if cut <= 0:
                break
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                out.append(sentence)
        return out

    def _find_cut(self):
        buf = self._buf
        # 1) 明确的句末标点：够一句话就切，早切才能早出声
        for i, ch in enumerate(buf):
            if ch in self.ENDINGS and i + 1 >= self.HARD_MIN_CHARS:
                return i + 1
        # 2) 已经攒得够长：逗号也允许切。语音助理里"早出声"比"句子完整"更重要，
        #    逗号处的停顿本来就很自然。
        if len(buf) >= self.soft_min_chars:
            for i in range(len(buf) - 1, self.HARD_MIN_CHARS - 1, -1):
                if buf[i] in self.SOFT:
                    return i + 1
        # 3) 太长则硬切
        if len(buf) >= self.max_chars:
            return self.max_chars
        return 0

    def flush(self):
        """模型输出结束：把剩下的也吐出来"""
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if rest else []

    def reset(self):
        self._buf = ""


# ======================================================================
# 分句流式播报
# ======================================================================

class SpeechPlayer:
    """
    逐句合成 + 播报，可被打断。

    生产者（LLM 回调线程）调 push() 塞句子；播报线程取出来合成并播放。
    打断：调 interrupt() 会立刻停播并清空待播队列。
    """

    def __init__(self, voice=None, rate=None, provider=None, api_key=None,
                 on_playing_changed=None, on_error=None,
                 max_chars=TTS_MAX_TEXT_CHARS):
        self.voice = voice or DEFAULT_LOCAL_VOICE
        self.rate = rate or DEFAULT_SPEECH_RATE
        self.provider = provider or DEFAULT_TTS_PROVIDER
        self.api_key = api_key
        self.on_playing_changed = on_playing_changed
        self.on_error = on_error
        self.max_chars = int(max_chars)

        self._q = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._thread = None
        self._stop = threading.Event()
        self._interrupted = threading.Event()
        self._producing_done = False
        self._done = threading.Event()
        self._seq = 0
        self._playing = False
        self.delay_gate_until = 0.0     # 在此之前不要开始播（给"即时应答音"让位）

        # 统计（用来量化首响延迟）
        self.first_audio_ms = None
        self.spoken_sentences = 0
        self.spoken_chars = 0

    # -------------------- 生产者接口 --------------------

    def begin(self):
        """开始一次回复的播报"""
        with self._lock:
            self._q = []
            self._producing_done = False
            self._done.clear()
            self._interrupted.clear()
            self._seq = 0
            self.first_audio_ms = None
            self.spoken_sentences = 0
            self.spoken_chars = 0
        self._t_begin = time.time()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def push(self, sentence):
        """塞一句待播报文本"""
        sentence = (sentence or "").strip()
        if not sentence or self._interrupted.is_set():
            return
        too_long = len(sentence) > self.max_chars
        if too_long:
            sentence = sentence[:self.max_chars]
        with self._cv:
            self._q.append(sentence)
            self._cv.notify()

    def end(self):
        """生产者结束（模型输出完了）"""
        with self._cv:
            self._producing_done = True
            self._cv.notify()

    def wait(self, timeout=120):
        """等播报结束（或被打算）"""
        return self._done.wait(timeout)

    # -------------------- 打断 --------------------

    def interrupt(self):
        """打断：立刻停播并清空队列"""
        self._interrupted.set()
        with self._cv:
            self._q = []
            self._cv.notify()
        self._stop_music()
        with self._lock:
            if self._producing_done or not self._q:
                self._done.set()

    def is_playing(self):
        return self._playing

    # -------------------- 播报线程 --------------------

    def _run(self):
        import asyncio
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except Exception as e:
            self._error("音频播放初始化失败: %s" % e)
            self._done.set()
            return

        from core.tts import TextToSound
        tts = TextToSound(provider=self.provider, voice=self.voice,
                          api_key=self.api_key, rate=self.rate)

        while not self._stop.is_set():
            with self._cv:
                while not self._q and not self._producing_done and not self._interrupted.is_set():
                    self._cv.wait(0.1)
                if self._interrupted.is_set():
                    break
                if not self._q and self._producing_done:
                    break
                sentence = self._q.pop(0)

            try:
                audio = asyncio.run(asyncio.wait_for(
                    tts.run(sentence), timeout=TTS_TIMEOUT_SECONDS))
            except Exception as e:
                self._error("语音合成失败: %s" % str(e)[:80])
                continue
            if not audio:
                self._error("语音合成返回空音频")
                continue
            if self._interrupted.is_set():
                break

            if self.first_audio_ms is None:
                self.first_audio_ms = (time.time() - self._t_begin) * 1000.0
            self.spoken_sentences += 1
            self.spoken_chars += len(sentence)
            self._play_and_wait(audio)

        self._set_playing(False)
        self._done.set()

    def _play_and_wait(self, audio_bytes):
        import pygame
        # 若刚播过"即时应答音"，等它说完再播正式内容，避免把它拦腰截断
        wait = self.delay_gate_until - time.time()
        while wait > 0 and not self._interrupted.is_set() and not self._stop.is_set():
            time.sleep(min(0.05, wait))
            wait = self.delay_gate_until - time.time()

        path = os.path.join(TEMP_DIR, "speak_%d.mp3" % (self._seq % 4))
        self._seq += 1
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            with open(path, "wb") as f:
                f.write(audio_bytes)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self._set_playing(True)
            # 播放期间持续检查打断
            while pygame.mixer.music.get_busy():
                if self._interrupted.is_set() or self._stop.is_set():
                    self._stop_music()
                    break
                time.sleep(0.05)
        except Exception as e:
            self._error("播放失败: %s" % str(e)[:80])
        finally:
            self._set_playing(False)

    def _stop_music(self):
        try:
            import pygame
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

    def _set_playing(self, playing):
        self._playing = playing
        if self.on_playing_changed:
            try:
                self.on_playing_changed(playing)
            except Exception:
                pass

    def _error(self, text):
        if self.on_error:
            try:
                self.on_error(text)
            except Exception:
                pass

    def close(self):
        self._stop.set()
        self.interrupt()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


# ======================================================================
# 对话引擎
# ======================================================================

class DialogueEngine:
    """
    把连续聆听、识别、大模型、分句流式播报串成一条链路。

    用法：
        eng = DialogueEngine(on_state=..., on_user_text=..., on_reply_chunk=...,
                             on_reply=..., on_subtitle=..., on_status=..., on_error=...)
        eng.start()
        eng.submit_text("你好")            # 文字输入
        eng.submit_audio(pcm)              # 也可以用音频（测试/文件输入）
        eng.stop()

    所有回调都在工作线程或采集线程里触发，界面层负责切回自己的线程。
    """

    def __init__(self, on_state=None, on_user_text=None, on_reply_chunk=None,
                 on_reply=None, on_subtitle=None, on_status=None, on_error=None,
                 api_key=None, voice=None, rate=None, system_prompt=None,
                 enable_listener=True, listener_kwargs=None):
        from core.stt import SoundToText
        from core.llm import LLM
        from core.continuous_listener import ContinuousListener

        self.on_state = on_state
        self.on_user_text = on_user_text
        self.on_reply_chunk = on_reply_chunk
        self.on_reply = on_reply
        self.on_subtitle = on_subtitle
        self.on_status = on_status
        self.on_error = on_error

        self.api_key = api_key
        self.stt = SoundToText(api_key=api_key)
        self.llm = LLM(api_key=api_key)
        if system_prompt:
            self.llm.system_prompt = system_prompt

        self.player = SpeechPlayer(
            voice=voice, rate=rate, api_key=api_key,
            on_playing_changed=self._on_playing_changed,
            on_error=lambda m: self._emit(self.on_error, m),
        )

        self._state = STATE_IDLE
        self._jobs = queue.Queue()
        self._stop = threading.Event()
        self._worker = None
        self._busy = False
        self.enable_web_search = False     # 默认不联网：老年场景先要快和稳
        self.searcher = None

        self.listener = None
        if enable_listener:
            kwargs = dict(listener_kwargs or {})
            kwargs.setdefault("on_status", lambda t: self._emit(self.on_status, t))
            self.listener = ContinuousListener(
                on_utterance=self._on_utterance,
                on_barge_in=self._on_barge_in,
                **kwargs)

        # 统计
        self.stats = {"asr_ms": 0.0, "llm_first_ms": 0.0, "first_audio_ms": 0.0,
                      "filler_ms": 0.0, "turns": 0, "barge_ins": 0, "confirmations": 0}

        # 确认式交互：等待用户对"关键信息复述"的答复
        self._pending_confirm = None
        self._pending_since = 0.0

        # 即时应答音（启动时后台预合成，播放延迟只有几十毫秒）
        self._fillers = []          # [(path, seconds), ...]
        self._filler_idx = 0
        self._filler_until = 0.0

    # -------------------- 即时应答音 --------------------

    def _prefetch_fillers(self):
        """
        预合成几句极短应答，缓存成文件。
        作用：识别成功后立刻出声，把"大模型首 token + 合成"的等待盖住。
        老年用户尤其需要这个即时反馈——否则他会以为机器没听见而重复说话。
        """
        if not TTS_FILLER_ENABLED:
            return
        try:
            import asyncio
            import pygame
            from core.tts import TextToSound
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            tts = TextToSound(provider=DEFAULT_TTS_PROVIDER, voice=self.player.voice,
                              api_key=self.api_key, rate=self.player.rate)
            for i, text in enumerate(TTS_FILLERS):
                audio = asyncio.run(asyncio.wait_for(tts.run(text),
                                                     timeout=TTS_TIMEOUT_SECONDS))
                if not audio:
                    continue
                path = os.path.join(TEMP_DIR, "filler_%d.mp3" % i)
                os.makedirs(TEMP_DIR, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(audio)
                try:
                    seconds = pygame.mixer.Sound(path).get_length()
                except Exception:
                    seconds = 1.2
                self._fillers.append((path, float(seconds)))
        except Exception as e:
            self._emit(self.on_status, "即时应答音预合成失败（不影响使用）：%s" % str(e)[:60])

    def _speak_filler(self):
        """播一句即时应答音（不阻塞，不等它说完）"""
        if not self._fillers:
            return
        try:
            import pygame
            path, seconds = self._fillers[self._filler_idx % len(self._fillers)]
            self._filler_idx += 1
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self._filler_until = time.time() + seconds + 0.15
            self.player.delay_gate_until = self._filler_until
            self.stats["filler_ms"] = 0.0
            # 应答音也是"我们在出声"，让监听器别把它当成用户说话
            if self.listener is not None:
                self.listener.set_playing(True)
        except Exception as e:
            self._emit(self.on_status, "应答音播放失败：%s" % str(e)[:60])

    # -------------------- 生命周期 --------------------

    def start(self, open_listener=True):
        """
        :param open_listener: 是否同时打开麦克风与回环。
                              False = 只用文字通道（没有麦克风、或用户只想打字时）
        """
        if self._worker and self._worker.is_alive():
            return True, ""
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        # 后台预合成即时应答音（不拖慢启动）
        if TTS_FILLER_ENABLED:
            threading.Thread(target=self._prefetch_fillers, daemon=True).start()
        if self.listener is not None and open_listener:
            ok, err = self.listener.start()
            if not ok:
                return False, err
            self._set_state(STATE_LISTENING)
        else:
            self._set_state(STATE_IDLE)
        return True, ""

    def stop(self):
        self._stop.set()
        self._jobs.put(None)
        if self.player:
            self.player.close()
        if self.listener is not None:
            self.listener.stop()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)

    def is_running(self):
        return bool(self._worker and self._worker.is_alive())

    # -------------------- 输入 --------------------

    def submit_text(self, text):
        """文字输入：走与语音完全相同的链路（也用于无麦克风环境的自检）"""
        text = (text or "").strip()
        if not text:
            return
        self._jobs.put(("text", text))

    def submit_audio(self, pcm):
        """直接喂一段 16k 单声道 PCM，当作"用户说的一句话"（测试用）"""
        if pcm:
            self._jobs.put(("audio", pcm))

    def _on_utterance(self, pcm):
        """采集线程回调：VAD 切出一句话"""
        if self._busy:
            # 正在处理上一轮：丢弃，避免任务堆积与"自己听自己"
            return
        self._jobs.put(("audio", pcm))

    def _on_barge_in(self):
        """采集线程回调：检测到用户插话"""
        self.stats["barge_ins"] += 1
        self._emit(self.on_status, "检测到插话，停止播报")
        if self.player:
            self.player.interrupt()

    def _on_playing_changed(self, playing):
        if self.listener is not None:
            self.listener.set_playing(playing)
        self._set_state(STATE_SPEAKING if playing else STATE_LISTENING)

    # -------------------- 确认式交互 --------------------

    def _needs_confirm(self, text):
        """识别结果里是否含关键信息（用药/金额/时间…），需要先复述确认"""
        if not CONFIRM_ENABLED:
            return None
        t = text or ""
        for kw in CONFIRM_KEYWORDS:
            if kw in t:
                return kw
        return None

    @staticmethod
    def _classify_answer(text):
        """
        判断用户对复述的答复：yes / no / other

        两条规则都是踩过坑才定下来的：
          1. **先完全匹配**：'没错' 必须在这里判为肯定。如果一上来做包含判断，
             它会被否定词里的 '错' 命中，把肯定听成否定（用药场景会直接误操作）。
          2. 再做包含判断时，**否定词优先**：'不对' 里含 '对'，顺序反了同样会判错。
        """
        t = "".join(ch for ch in (text or "") if ch not in "，。！？、,.!? \t")
        if not t or len(t) > CONFIRM_MAX_ANSWER_CHARS:
            return "other"
        for w in CONFIRM_YES:
            if t == w:
                return "yes"
        for w in CONFIRM_NO:
            if t == w:
                return "no"
        for w in CONFIRM_NO:
            if w in t:
                return "no"
        for w in CONFIRM_YES:
            if w in t:
                return "yes"
        return "other"

    def _speak_once(self, text):
        """单独播一句话（确认提示等），复用分句播报链路，因此也会正确地进入/退出播报态"""
        try:
            self.player.begin()
            self.player.push(text)
            self.player.end()
            self.player.wait(timeout=30)
        except Exception as e:
            self._emit(self.on_error, "播报失败: %s" % str(e)[:80])

    # -------------------- 工作线程 --------------------

    def _worker_loop(self):
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None:
                break
            kind, payload = job
            self._busy = True
            try:
                if kind == "audio":
                    text = self._transcribe(payload)
                else:
                    text = payload
                if not text:
                    self._set_state(STATE_LISTENING)
                    continue

                # ---- 先处理"上一轮关键信息的答复" ----
                if self._pending_confirm:
                    answer = self._classify_answer(text)
                    pending = self._pending_confirm
                    expired = (time.time() - self._pending_since) > CONFIRM_TIMEOUT_SECONDS
                    self._pending_confirm = None
                    if not expired and answer == "yes":
                        text = pending          # 用确认过的原文继续，避免拿错字去回答
                        self._emit(self.on_status, "已确认，按识别结果继续")
                    elif not expired and answer == "no":
                        self._emit(self.on_status, "用户否定了识别结果")
                        self._emit(self.on_subtitle, ("系统", "好，请您再说一次"))
                        self._speak_once("好的，请您再说一次。")
                        continue
                    # 超时或答非所问：当作新的一句话正常往下走

                self._emit(self.on_user_text, text)
                self._emit(self.on_subtitle, ("用户", text))

                # ---- 关键信息先复述确认（只对语音输入做；打字输入是用户自己敲的，不必问）----
                if kind == "audio" and self._needs_confirm(text):
                    self._pending_confirm = text
                    self._pending_since = time.time()
                    prompt = "您是说：%s，对吗？" % text
                    self._emit(self.on_status, "关键信息，先复述确认")
                    self._emit(self.on_subtitle, ("系统", prompt))
                    self.stats["confirmations"] += 1
                    self._speak_once(prompt)
                    continue

                # 语音输入才播"即时应答音"：打字输入的用户不需要被安抚
                if kind == "audio":
                    t_fill = time.time()
                    self._speak_filler()
                    self.stats["filler_ms"] = (time.time() - t_fill) * 1000.0
                reply = self._respond(text)
                if reply:
                    self._emit(self.on_reply, reply)
                    self._emit(self.on_subtitle, ("助理", reply))
                    self.stats["turns"] += 1
            except Exception as e:
                self._emit(self.on_error, "对话处理异常: %s" % str(e)[:120])
            finally:
                self._busy = False
                # 兜底：任何异常路径都必须回到"聆听"，否则监听器会一直以为在播报、
                # VAD 被永久关掉（这类"状态卡死"最难查）
                if self.listener is not None:
                    self.listener.set_playing(False)
                if not self._stop.is_set():
                    self._set_state(STATE_LISTENING)

    def _transcribe(self, pcm):
        """识别 + 超时重试（实测 ASR 延迟抖动可达 60 秒，必须兜住）"""
        self._set_state(STATE_RECOGNIZING)
        t0 = time.time()
        last_err = ""
        for attempt in range(ASR_MAX_RETRIES + 1):
            if self._stop.is_set():
                return ""
            wav = pcm_to_wav_bytes(pcm)
            text, err = self.stt.transcribe(wav)
            if not err and text:
                self.stats["asr_ms"] = (time.time() - t0) * 1000.0
                return text.strip()
            last_err = err or "识别结果为空"
            if attempt < ASR_MAX_RETRIES:
                self._emit(self.on_status, "识别超时或失败，正在重试…")
        self._emit(self.on_error, "语音识别失败：%s" % last_err)
        self._emit(self.on_subtitle, ("系统", "没听清，请再说一次"))
        return ""

    def _respond(self, text):
        """大模型流式回复 + 分句流式播报"""
        self._set_state(STATE_THINKING)
        splitter = SentenceSplitter()
        self.player.begin()
        t0 = time.time()
        first_token_seen = [False]

        def on_chunk(tok):
            if not first_token_seen[0]:
                first_token_seen[0] = True
                self.stats["llm_first_ms"] = (time.time() - t0) * 1000.0
            self._emit(self.on_reply_chunk, tok)
            for sentence in splitter.feed(tok):
                self.player.push(sentence)

        context = ""
        if self.enable_web_search:
            context = self._web_search(text)

        reply, err = self.llm.chat_stream(text, on_chunk=on_chunk, context=context,
                                          concise_hint=None)
        for sentence in splitter.flush():
            self.player.push(sentence)
        self.player.end()
        self.player.wait(timeout=180)
        self.stats["first_audio_ms"] = self.player.first_audio_ms or 0.0
        if err:
            self._emit(self.on_error, err)
        return reply

    def _web_search(self, text):
        try:
            if self.searcher is None:
                from core.searcher import WebSearch
                self.searcher = WebSearch()
            results, err = self.searcher.search(text)
            if err or not results:
                return ""
            return "\n".join("%d. %s %s\n   %s" % (i, r.get("title", ""), r.get("url", ""),
                                                  r.get("snippet", ""))
                             for i, r in enumerate(results, 1))
        except Exception:
            return ""

    # -------------------- 状态与事件 --------------------

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self._emit(self.on_state, state)

    @property
    def state(self):
        return self._state

    def state_label(self):
        return STATE_LABELS.get(self._state, self._state)

    @staticmethod
    def _emit(cb, *args):
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            pass


# ========================= 测试入口 =========================
if __name__ == "__main__":
    print("=== 连续对话引擎：无麦克风自检（文字输入 + 分句流式播报）===")
    print("注意：会调用真实 API 并出声播报。仅用于开发自检。")
    print("用法：python -m core.dialogue --text \"你好\"   文字输入自检")
    print("      python -m core.dialogue --wav temp/xxx.wav  音频输入自检（含即时应答音）")
    print("      不带参数则只做离线分句自检")

    import sys
    if "--text" in sys.argv or "--wav" in sys.argv:
        mode = "audio" if "--wav" in sys.argv else "text"
        if mode == "audio":
            wav_path = sys.argv[sys.argv.index("--wav") + 1]
            pcm = read_wav_to_pcm(wav_path)
            print("  读入音频 %s：%.1f 秒" % (wav_path, len(pcm) / 2.0 / RECORD_RATE))
        else:
            question = sys.argv[sys.argv.index("--text") + 1]
        events = []
        eng = DialogueEngine(
            enable_listener=False,
            on_state=lambda s: (events.append(("state", s)), print("  状态:", s, flush=True)),
            on_user_text=lambda t: print("  用户:", t, flush=True),
            on_reply_chunk=lambda t: print(t, end="", flush=True),
            on_reply=lambda r: print("\n  完整回复 %d 字" % len(r), flush=True),
            on_subtitle=lambda who, t: print("  字幕[%s] %s" % (who, t[:40]), flush=True),
            on_status=lambda t: print("  状态消息:", t, flush=True),
            on_error=lambda t: print("  错误:", t, flush=True),
        )
        ok, err = eng.start()
        if not ok:
            print("启动失败:", err)
            sys.exit(1)
        # 等即时应答音预合成完成，否则第一轮享受不到它
        print("  （等待即时应答音预合成…）", flush=True)
        time.sleep(4.0)
        t_submit = time.time()
        if mode == "audio":
            eng.submit_audio(pcm)
        else:
            eng.submit_text(question)
        time.sleep(1.0)
        deadline = time.time() + 90
        while eng._busy and time.time() < deadline:
            time.sleep(0.2)
        time.sleep(0.5)
        print("\n=== 统计 ===")
        for k, v in eng.stats.items():
            print("  %-14s %s" % (k, ("%.0f ms" % v) if k.endswith("_ms") else v))
        print("  应答音已缓存   %d 句" % len(eng._fillers))
        if eng.stats["first_audio_ms"]:
            total = eng.stats["asr_ms"] + eng.stats["first_audio_ms"]
            if mode == "audio":
                total += eng.stats.get("filler_ms", 0.0)
            print("  从提交到首次出声 约 %.0f ms" % total)
        eng.stop()
    else:
        # 离线自检分句器（不联网）
        sp = SentenceSplitter()
        print("\n分句自检:")
        for tok in ["今天", "天气", "不错。", "气温", "二十度，", "记得", "带伞！"]:
            for s in sp.feed(tok):
                print("   切出:", s)
        for s in sp.flush():
            print("   收尾:", s)

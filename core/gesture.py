# -*- coding: utf-8 -*-
"""
手势识别（MediaPipe 手部 21 关键点 + 几何规则）

【为什么用几何规则，而不是训练一个分类模型】
  1. 静态手势的判据本来就是几何量：哪几根手指伸直、拇指朝向哪边；
  2. 不需要数据集与训练，改词表不用重训、可解释、出错能查；
  3. **可以离线自检**——用合成关键点就能验证分类器，不必对着摄像头反复试。

【诚实的边界】
  完整手语是"手形 + 位置 + 运动 + 面部表情"的语法系统，全量识别属研究级难题。
  本模块只做**受限词表的静态手势**（外加"挥手"这一种运动手势），
  用于两类场景：
    · 控制程序（control）：✋开始/停止聆听、✊取消、👍确认、👎停止播报、
                         ✌️切换字幕、挥手紧急停止
    · 代替说话（speak）  ：把手势识别成短语并用语音播报给对方听，面向聋哑用户
  "手势 -> 说什么" 是可配置词表（config.GESTURE_PHRASES），不是标准手语。

【关键点编号（MediaPipe Hands）】
  0 腕；1-4 拇指；5-8 食指；9-12 中指；13-16 无名指；17-20 小指
  每根手指取 MCP / PIP / DIP / TIP 四点。
"""

import math
import os
import threading
import time

from config import (
    HAND_MODEL_PATH, HAND_MAX_HANDS, HAND_DETECT_CONFIDENCE,
    GESTURE_HOLD_MS, GESTURE_COOLDOWN_MS, GESTURE_MIN_CONFIDENCE,
    GESTURE_SAMPLE_INTERVAL_MS, GESTURE_MODE,
    GESTURE_PHRASES, GESTURE_CONTROLS, GESTURE_CONTROL_LABELS,
)

# 关键点索引
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# 静态手势表：手指伸展模式（拇指, 食指, 中指, 无名指, 小指）-> 手势名
# 数字手势采用中国习惯：1 食指 / 2 食指+中指 / 3 拇指+食指+中指 /
# 4 四指（不含拇指）/ 5 五指张开 / 6 拇指+小指 / 8 拇指+食指（"手枪"）
STATIC_PATTERNS = {
    (1, 1, 1, 1, 1): "open_palm",
    (0, 0, 0, 0, 0): "fist",
    (0, 1, 0, 0, 0): "point",
    (0, 1, 1, 0, 0): "two",
    (1, 1, 1, 0, 0): "three",
    (0, 1, 1, 1, 1): "four",
    (1, 0, 0, 0, 1): "six",
    (1, 1, 0, 0, 0): "eight",
}
# 拇指单独伸出：需要靠朝向区分"赞/踩"
THUMB_ONLY = (1, 0, 0, 0, 0)


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GestureClassifier:
    """
    从 21 个手部关键点判断静态手势。

    判据（都可解释、可调）：
      手指伸直：指尖到手腕的距离 明显大于 该指 PIP 到手腕的距离
      拇指伸直：拇指尖到小指根的距离 明显大于 拇指根到小指根的距离
                （拇指收拢时会贴近掌心，也就靠近小指根）
    """

    def __init__(self, extend_ratio=1.05, thumb_ratio=1.20):
        self.extend_ratio = extend_ratio
        self.thumb_ratio = thumb_ratio

    def finger_states(self, lm):
        """返回 (thumb, index, middle, ring, pinky) 的 0/1 伸展状态"""
        wrist = lm[WRIST]
        states = {}
        for name, (_mcp, pip, _dip, tip) in FINGERS.items():
            states[name] = 1 if _dist(lm[tip], wrist) > _dist(lm[pip], wrist) * self.extend_ratio else 0
        # 拇指：用"到小指根的距离"判断收拢/展开
        thumb_ext = _dist(lm[THUMB_TIP], lm[PINKY_MCP]) > \
            _dist(lm[THUMB_MCP], lm[PINKY_MCP]) * self.thumb_ratio
        return (1 if thumb_ext else 0, states["index"], states["middle"],
                states["ring"], states["pinky"])

    def classify(self, lm):
        """
        :param lm: 21 个 (x, y, z) 关键点（归一化坐标）
        :return: (手势名 或 None, 置信度 0~1)
        """
        if not lm or len(lm) < 21:
            return None, 0.0
        pattern = self.finger_states(lm)

        # 拇指单独伸出：靠拇指尖相对手腕的高低区分"赞"与"踩"
        # （图像坐标 y 向下增大，拇指尖 y 更小 = 朝上）
        if pattern == THUMB_ONLY:
            up = lm[THUMB_TIP][1] < lm[WRIST][1]
            return ("thumbs_up" if up else "thumbs_down"), 0.9

        name = STATIC_PATTERNS.get(pattern)
        if name is None:
            return None, 0.0
        # 置信度：以"五根手指的伸展判定清晰度"粗略估计
        margin = self._margin(lm)
        conf = max(0.5, min(0.99, 0.6 + margin))
        return name, conf

    def _margin(self, lm):
        """手指伸直/弯曲的判定裕度，用来粗略反映这次判定有多"干脆" """
        wrist = lm[WRIST]
        margins = []
        for _name, (_mcp, pip, _dip, tip) in FINGERS.items():
            d_tip = _dist(lm[tip], wrist)
            d_pip = _dist(lm[pip], wrist) + 1e-6
            margins.append(abs(d_tip / d_pip - self.extend_ratio))
        return min(0.35, sum(margins) / len(margins))


class WaveDetector:
    """挥手检测：张开手掌 + 腕部横向来回移动（唯一用到的运动手势）"""

    def __init__(self, window=8, min_swings=2, min_amplitude=0.04):
        self.window = window
        self.min_swings = min_swings
        self.min_amplitude = min_amplitude
        self._xs = []

    def update(self, gesture, wrist_x):
        """喂入当前手势与腕部横坐标，返回是否检测到挥手"""
        if gesture != "open_palm":
            self._xs = []
            return False
        self._xs.append(wrist_x)
        if len(self._xs) > self.window:
            self._xs.pop(0)
        if len(self._xs) < self.window:
            return False
        swings = 0
        for i in range(1, len(self._xs)):
            dx = self._xs[i] - self._xs[i - 1]
            if abs(dx) >= self.min_amplitude / 2:
                swings += 1
        # 方向改变次数达到阈值且总幅度够大
        direction_changes = 0
        last = 0
        for i in range(1, len(self._xs)):
            dx = self._xs[i] - self._xs[i - 1]
            if abs(dx) < self.min_amplitude / 4:
                continue
            d = 1 if dx > 0 else -1
            if last != 0 and d != last:
                direction_changes += 1
            last = d
        amplitude = max(self._xs) - min(self._xs)
        return direction_changes >= self.min_swings and amplitude >= self.min_amplitude

    def reset(self):
        self._xs = []


class GestureStabilizer:
    """
    防抖：同一手势必须**稳定保持** hold_ms 才触发一次，触发后 cooldown_ms 内不再重复。
    没有它，手在摄像头前稍微一晃就会连续触发。
    """

    def __init__(self, hold_ms=GESTURE_HOLD_MS, cooldown_ms=GESTURE_COOLDOWN_MS,
                 min_confidence=GESTURE_MIN_CONFIDENCE):
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.min_confidence = min_confidence
        self._current = None
        self._since = 0.0
        self._fired_at = 0.0

    def update(self, gesture, confidence, now=None):
        """
        :return: 本次应当触发的手势名（None 表示不触发）
        """
        now = now if now is not None else time.time() * 1000.0
        if gesture is None or confidence < self.min_confidence:
            self._current = None
            return None
        if gesture != self._current:
            self._current = gesture
            self._since = now
            return None
        # 冷却判断必须放行"从未触发过"的情况：
        # _fired_at 初值为 0，若直接比 (now - 0) >= cooldown，
        # 首次触发会被自己的冷却逻辑永久挡住（实测表现为"手势完全没反应"）。
        cooled = (self._fired_at <= 0.0) or ((now - self._fired_at) >= self.cooldown_ms)
        if (now - self._since) >= self.hold_ms and cooled:
            self._fired_at = now
            return gesture
        return None


class HandTracker:
    """
    摄像头 + MediaPipe HandLandmarker，持续输出手势事件。

    用法：
        tracker = HandTracker(on_gesture=lambda name, conf: ...)
        tracker.start()      # 打开摄像头
        ...
        tracker.stop()

    模型文件缺失时会明确报错并告诉你怎么下载（不静默失败）。
    """

    def __init__(self, on_gesture=None, on_status=None, on_error=None,
                 mode=None, model_path=HAND_MODEL_PATH):
        self.on_gesture = on_gesture
        self.on_status = on_status
        self.on_error = on_error
        self.mode = mode or GESTURE_MODE
        self.model_path = model_path
        self.classifier = GestureClassifier()
        self.wave = WaveDetector()
        self.stabilizer = GestureStabilizer()
        self._stop = threading.Event()
        self._thread = None
        self.camera_index = 0
        self.stats = {"frames": 0, "hands": 0, "events": 0}
        self.last_gesture = None

    # -------------------- 生命周期 --------------------

    def start(self):
        if not os.path.exists(self.model_path):
            msg = ("缺少手部模型文件：%s\n"
                   "请下载（约 7.5MB）：%s" % (self.model_path, self.model_url_hint()))
            self._emit(self.on_error, msg)
            return False, msg
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True, ""

    def model_url_hint(self):
        from config import HAND_MODEL_URL
        return HAND_MODEL_URL

    def stop(self, timeout=3.0):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def set_mode(self, mode):
        """切换"手势用于控制" / "手势用于说话" """
        self.mode = mode

    # -------------------- 采集与识别 --------------------

    def _run(self):
        try:
            # 压掉 MediaPipe/TFLite 往 stderr 打的大量 INFO（XNNPACK、feedback manager 等），
            # 否则控制台全是噪音，真正的错误反而看不见
            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
                num_hands=HAND_MAX_HANDS,
                min_hand_detection_confidence=HAND_DETECT_CONFIDENCE)
            landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            self._emit(self.on_error, "初始化手部识别失败：%s" % str(e)[:150])
            return

        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self._emit(self.on_error, "无法打开摄像头（可能被占用或没有权限）")
                return
            self._emit(self.on_status, "手势识别已启动（摄像头开启）")
            interval = max(0.03, GESTURE_SAMPLE_INTERVAL_MS / 1000.0)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(interval)
                    continue
                self.stats["frames"] += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    result = landmarker.detect(mp_image)
                except Exception:
                    time.sleep(interval)
                    continue

                gesture, conf = None, 0.0
                if result.hand_landmarks:
                    self.stats["hands"] += 1
                    lm = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
                    gesture, conf = self.classifier.classify(lm)
                    if gesture == "open_palm" and self.wave.update("open_palm", lm[WRIST][0]):
                        gesture, conf = "wave", 0.9
                    else:
                        self.wave.update(gesture, lm[WRIST][0])
                else:
                    self.wave.reset()

                fired = self.stabilizer.update(gesture, conf)
                if fired:
                    self.stats["events"] += 1
                    self.last_gesture = fired
                    self._emit(self.on_gesture, fired, conf)
                time.sleep(interval)
        except Exception as e:
            self._emit(self.on_error, "手势识别异常：%s" % str(e)[:120])
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            self._emit(self.on_status, "手势识别已停止（摄像头已释放）")

    def _emit(self, cb, *args):
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            pass

    # -------------------- 词表 --------------------

    @staticmethod
    def phrase_for(gesture):
        """说话模式：手势 -> 要说出的话（未收录返回 None）"""
        return GESTURE_PHRASES.get(gesture)

    @staticmethod
    def action_for(gesture):
        """控制模式：手势 -> 动作名（未收录返回 None）"""
        return GESTURE_CONTROLS.get(gesture)

    @staticmethod
    def action_label(action):
        return GESTURE_CONTROL_LABELS.get(action, action)


# ======================================================================
# 自检：用合成关键点验证分类器（不需要摄像头）
# ======================================================================

def make_hand(thumb=0, index=0, middle=0, ring=0, pinky=0, thumb_up=True):
    """
    合成一只手的 21 个关键点。

    构造思路（模仿真实几何关系）：
      腕在原点；四指根沿 x 排开；手指伸直时指尖远离腕，弯曲时收回到掌内。
      拇指根在 (0, -0.3)，伸直时拇指尖向外伸远，收拢时贴近小指根。
    """
    lm = [(0.0, 0.0, 0.0)] * 21
    lm = list(lm)
    lm[WRIST] = (0.5, 0.6, 0.0)
    # 四指：MCP 在掌上，PIP/DIP/TIP 依次向外（伸直）或折回（弯曲）
    base_x = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.61}
    ext = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, x in base_x.items():
        mcp_i, pip_i, dip_i, tip_i = FINGERS[name]
        mcp = (x, 0.45, 0.0)
        lm[mcp_i] = mcp
        if ext[name]:
            lm[pip_i] = (x, 0.33, 0.0)
            lm[dip_i] = (x, 0.24, 0.0)
            lm[tip_i] = (x, 0.16, 0.0)
        else:
            # 弯曲：指尖折回掌内，离腕更近
            lm[pip_i] = (x, 0.36, 0.0)
            lm[dip_i] = (x, 0.44, 0.0)
            lm[tip_i] = (x, 0.52, 0.0)
    # 拇指
    lm[THUMB_CMC] = (0.40, 0.55, 0.0)
    lm[THUMB_MCP] = (0.34, 0.50, 0.0)
    if thumb:
        lm[THUMB_IP] = (0.28, 0.42 if thumb_up else 0.62, 0.0)
        lm[THUMB_TIP] = (0.22 if thumb_up else 0.26, 0.30 if thumb_up else 0.74, 0.0)
    else:
        # 收拢：贴近掌心/小指根
        lm[THUMB_IP] = (0.44, 0.52, 0.0)
        lm[THUMB_TIP] = (0.50, 0.50, 0.0)
    return lm


if __name__ == "__main__":
    print("=== 手势分类器自检（合成关键点，不需要摄像头）===")
    clf = GestureClassifier()
    cases = [
        ("张开手掌 open_palm", dict(thumb=1, index=1, middle=1, ring=1, pinky=1), "open_palm"),
        ("握拳 fist", dict(thumb=0, index=0, middle=0, ring=0, pinky=0), "fist"),
        ("拇指向上 thumbs_up", dict(thumb=1, index=0, middle=0, ring=0, pinky=0, thumb_up=True), "thumbs_up"),
        ("拇指向下 thumbs_down", dict(thumb=1, index=0, middle=0, ring=0, pinky=0, thumb_up=False), "thumbs_down"),
        ("食指 point", dict(thumb=0, index=1, middle=0, ring=0, pinky=0), "point"),
        ("二 two", dict(thumb=0, index=1, middle=1, ring=0, pinky=0), "two"),
        ("三 three", dict(thumb=1, index=1, middle=1, ring=0, pinky=0), "three"),
        ("四 four", dict(thumb=0, index=1, middle=1, ring=1, pinky=1), "four"),
        ("六 six", dict(thumb=1, index=0, middle=0, ring=0, pinky=1), "six"),
        ("八 eight", dict(thumb=1, index=1, middle=0, ring=0, pinky=0), "eight"),
    ]
    ok = True
    for label, kwargs, want in cases:
        lm = make_hand(**kwargs)
        got, conf = clf.classify(lm)
        good = (got == want)
        ok = ok and good
        print("  %s %-22s -> %-12s 置信度 %.2f" % ("OK " if good else "×  ", label, got, conf))

    print("\n=== 防抖（稳定保持才触发）===")
    st = GestureStabilizer(hold_ms=600, cooldown_ms=1500, min_confidence=0.6)
    t = 0.0
    fired_at = []
    for i in range(12):                 # 1.2 秒内持续保持 open_palm
        t += 100.0
        f = st.update("open_palm", 0.9, now=t)
        if f:
            fired_at.append(int(t))
    print("  持续保持 1.2 秒，触发时刻(ms): %s（期望只在 600ms 后触发一次）" % fired_at)
    ok2 = fired_at == [700] or fired_at == [600]
    print("  判定: %s" % ("通过" if ok2 else "不通过"))

    print("\n=== 挥手检测 ===")
    wd = WaveDetector()
    got_wave = False
    for i, x in enumerate([0.50, 0.56, 0.44, 0.58, 0.42, 0.57, 0.43, 0.55]):
        if wd.update("open_palm", x):
            got_wave = True
    print("  左右来回移动 -> 挥手 %s" % ("检出" if got_wave else "未检出"))
    wd2 = WaveDetector()
    steady = any(wd2.update("open_palm", 0.50) for _ in range(8))
    print("  手掌静止   -> 挥手 %s（应不检出）" % ("误检出" if steady else "未检出"))
    ok3 = got_wave and not steady
    print("  判定: %s" % ("通过" if ok3 else "不通过"))

    print("\n=== 词表（说话模式 / 控制模式）===")
    for g in ("open_palm", "fist", "thumbs_up", "thumbs_down", "point", "two", "six"):
        print("  %-12s 说话:%-10s 控制:%s"
              % (g, HandTracker.phrase_for(g) or "-",
                 HandTracker.action_label(HandTracker.action_for(g)) if HandTracker.action_for(g) else "-"))

    print("\n=== 汇总: 分类 %s | 防抖 %s | 挥手 %s ==="
          % ("通过" if ok else "不通过", "通过" if ok2 else "不通过", "通过" if ok3 else "不通过"))

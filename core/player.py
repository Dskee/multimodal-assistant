# -*- coding: utf-8 -*-
"""
音频播放模块
使用 pygame.mixer 播放 MP3 音频字节
由于 pygame.mixer.Sound 对 MP3 支持不可靠，
我们采用"先写临时文件 -> 用 music.load/play 播放"的方式
"""

import os
import threading
import time

import pygame

from config import TEMP_REPLY_FILE


class AudioPlayer:
    """
    音频播放器
    用法：
        player = AudioPlayer()
        player.play(audio_bytes, on_finished=lambda: print("播放结束"))
    """

    def __init__(self):
        # 初始化 pygame.mixer（频率 22050 适合语音）
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        self._playing = False           # 是否正在播放
        self._monitor_thread = None     # 监控播放状态的线程
        self._on_finished = None        # 播放完成回调
        self._temp_path = TEMP_REPLY_FILE

    def play(self, audio_bytes, on_finished=None):
        """
        播放音频
        :param audio_bytes: MP3 格式的音频字节
        :param on_finished: 播放完成后的回调函数（无参数）
        :return: 是否成功开始播放
        """
        if self._playing:
            # 如果正在播放，先停止
            self.stop()

        # 确保临时目录存在
        os.makedirs(os.path.dirname(self._temp_path), exist_ok=True)

        try:
            # 0. 释放上一次播放占用的文件句柄（Windows 上 pygame.mixer.music
            #    播放完不会自动关闭文件，导致 temp/reply.mp3 被占用、
            #    下一次写入 Permission denied 而无法播放）
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

            # 1. 把音频字节写入临时文件
            with open(self._temp_path, "wb") as f:
                f.write(audio_bytes)

            # 2. 加载并播放
            pygame.mixer.music.load(self._temp_path)
            pygame.mixer.music.play()

            self._playing = True
            self._on_finished = on_finished

            # 3. 启动后台线程监控播放是否结束
            self._monitor_thread = threading.Thread(target=self._monitor_playback, daemon=True)
            self._monitor_thread.start()

            return True

        except Exception as e:
            print(f"播放失败: {e}")
            self._cleanup_temp()
            return False

    def stop(self):
        """停止播放"""
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            pygame.mixer.music.unload()  # 释放文件句柄
        except Exception:
            pass
        self._playing = False

    def is_playing(self):
        """返回是否正在播放"""
        return self._playing

    def _monitor_playback(self):
        """
        后台监控循环：检测 pygame.mixer.music.get_busy() 返回 False 时说明播放结束
        """
        while self._playing:
            if not pygame.mixer.music.get_busy():
                # 播放结束
                self._playing = False
                try:
                    pygame.mixer.music.unload()  # 释放文件句柄，允许下次覆盖写入
                except Exception:
                    pass
                self._cleanup_temp()

                # 调用回调
                if self._on_finished:
                    try:
                        self._on_finished()
                    except Exception as e:
                        print(f"播放完成回调出错: {e}")
                break

            time.sleep(0.1)  # 每 100ms 检查一次

    def _cleanup_temp(self):
        """删除临时音频文件"""
        try:
            if os.path.exists(self._temp_path):
                os.remove(self._temp_path)
        except Exception:
            pass


# ========================= 测试入口 =========================
if __name__ == "__main__":
    # 需要先生成 test_tts.mp3
    import asyncio
    from tts import TextToSound

    async def test():
        print("正在合成语音...")
        tts = TextToSound()
        audio = await tts.run("这是一个音频播放测试，听到这段话说明一切正常")
        if audio:
            player = AudioPlayer()
            print("开始播放...")
            player.play(audio, on_finished=lambda: print("播放结束！"))

            # 阻塞主线程等待播放完成
            while player.is_playing():
                time.sleep(0.2)
        else:
            print("TTS 合成失败")

    asyncio.run(test())

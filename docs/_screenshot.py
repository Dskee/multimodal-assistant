# -*- coding: utf-8 -*-
"""离屏渲染主窗口，出一张 README 用的截图。

不弹窗、不碰麦克风和摄像头：QT_QPA_PLATFORM=offscreen 让 Qt 画到内存里，
窗口构造完直接 grab() 存 PNG。构造失败就说明界面依赖真机设备，那就不要截图。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
from ui.main_window import MainWindow  # noqa: E402

window = MainWindow()
window.resize(1180, 760)
window.show()
app.processEvents()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "main-window.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
pix = window.grab()
pix.save(out, "PNG")
print("saved", out, pix.width(), "x", pix.height())
# -*- coding: utf-8 -*-
"""
程序入口：创建 QApplication + 主窗口
直接运行此文件即可启动应用
"""

import sys
import os

# 确保能找到项目根目录与 libs（内置 typing_extensions，修复 openai 导入问题）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    # 高 DPI 支持：让界面随系统显示缩放比例正确放大（高分屏下字不再又小又糊）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 创建 QApplication（PyQt5 必须且只能有一个）
    app = QApplication(sys.argv)

    # 创建主窗口并显示
    window = MainWindow()
    window.show()

    # 进入事件循环，直到窗口关闭
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

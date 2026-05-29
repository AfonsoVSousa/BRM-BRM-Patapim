# -*- coding: utf-8 -*-
"""
main.py — Entry point for EMG → Robot Dog Controller.
Run: python main.py
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui import EMGRobotWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = EMGRobotWindow()
    w.show()
    sys.exit(app.exec_())

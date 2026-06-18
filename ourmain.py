# ============================================================
# FILE: ourmain.py
#
# Launches the EMG Robot Dog GUI and loads the Qt application
# entry point for the desktop control workflow.
# ============================================================

import ctypes
ctypes.cdll.LoadLibrary(r"C:\Windows\System32\nicaiu.dll")

import sys
from PyQt5.QtWidgets import QApplication
from ui import EMGRobotWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = EMGRobotWindow()
    w.show()
    sys.exit(app.exec_())

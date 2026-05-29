# -*- coding: utf-8 -*-
"""
ui.py — CalibPanel widget + EMGRobotWindow main window.

Changes vs. original:
  • View 3 (Camera + EMG): 4 normalised plots on the left, live camera on the
    right, and an action-status box below the camera.
  • Bilateral masseter contraction tracking feeds the new gesture logic.
  • emg_to_robot() now receives bilateral state and returns independently of
    whether frontalis is active (turning is decoupled from forward movement).
  • Legend updated to reflect new gesture mapping.
"""

import time
import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QFrame, QStackedWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import (
    FS, WINDOW_SEC, WINDOW_SAMPLES, CHUNK,
    MUSCLES, COLORS,
    FRONTALIS_NORM_THRESHOLD, MASSETER_DIFF_THRESHOLD,
    POWER_CONTRACTIONS,
    DANCE_BILATERAL_CONTRACTIONS, DANCE_FRONTALIS_CONTRACTIONS,
    BUZZER_BILATERAL_CONTRACTIONS,
    lp_a, ROBOT_IP,
)
from emg_core         import MuscleCalib, generate_signals, process_chunk
from robot_controller import RobotDog, emg_to_robot


# ============================================================
# CALIBRATION PANEL (one per muscle) — unchanged
# ============================================================

class CalibPanel(QFrame):

    def __init__(self, calib: MuscleCalib):
        super().__init__()
        self.calib = calib
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {calib.color}44; "
            f"border-radius: 6px; padding: 4px; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(3)

        title = QLabel(calib.name)
        title.setStyleSheet(
            f"color: {calib.color}; font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self.rest_btn = QPushButton("▶  Start REST calibration")
        self.rest_btn.setCheckable(True)
        self.rest_btn.setStyleSheet(self._btn_style("#4a90d9"))
        self.rest_btn.clicked.connect(self.toggle_rest)
        layout.addWidget(self.rest_btn)

        self.max_btn = QPushButton("▶  Start MAX calibration")
        self.max_btn.setCheckable(True)
        self.max_btn.setStyleSheet(self._btn_style("#e07b39"))
        self.max_btn.clicked.connect(self.toggle_max)
        layout.addWidget(self.max_btn)

        self.status = QLabel("Not calibrated")
        self.status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.counter = QLabel("")
        self.counter.setStyleSheet("color: #ffcc02; font-size: 10px;")
        layout.addWidget(self.counter)

        # Threshold slider
        thr_row = QHBoxLayout()
        thr_lbl = QLabel("Threshold:")
        thr_lbl.setStyleSheet("color: #ccc; font-size: 10px;")
        self.thr_slider = QSlider(Qt.Horizontal)
        self.thr_slider.setMinimum(0)
        self.thr_slider.setMaximum(100)
        self.thr_slider.setValue(int(calib.threshold * 100))
        self.thr_val_lbl = QLabel(f"{calib.threshold:.2f}")
        self.thr_val_lbl.setStyleSheet(
            "color: #00ff88; font-size: 10px; min-width: 32px;")
        self.thr_slider.valueChanged.connect(self._on_threshold_changed)
        thr_row.addWidget(thr_lbl)
        thr_row.addWidget(self.thr_slider)
        thr_row.addWidget(self.thr_val_lbl)
        layout.addLayout(thr_row)

        self.contraction_lbl = QLabel("Contractions: 0")
        self.contraction_lbl.setStyleSheet(
            "color: #00ff88; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.contraction_lbl)

    def _btn_style(self, color):
        return (f"QPushButton {{ background: {color}33; color: #ccc; "
                f"border: 1px solid {color}88; border-radius: 4px; "
                f"padding: 4px; font-size: 10px; }}"
                f"QPushButton:checked {{ background: {color}99; color: white; }}"
                f"QPushButton:hover {{ background: {color}55; }}")

    def toggle_rest(self):
        if self.rest_btn.isChecked():
            self.calib.start_rest()
            self.rest_btn.setText("■  Stop REST calibration")
        else:
            self.calib.stop_rest()
            self.rest_btn.setText("▶  Start REST calibration")

    def toggle_max(self):
        if self.max_btn.isChecked():
            self.calib.start_max()
            self.max_btn.setText("■  Stop MAX calibration")
        else:
            self.calib.stop_max()
            self.max_btn.setText("▶  Start MAX calibration")

    def _on_threshold_changed(self, v):
        self.calib.threshold = v / 100.0
        self.thr_val_lbl.setText(f"{self.calib.threshold:.2f}")

    def update_counter(self):
        if self.calib.cal_rest:
            self.counter.setText(f"REST samples: {len(self.calib.rest_buf)}")
        elif self.calib.cal_max:
            self.counter.setText(f"MAX samples: {len(self.calib.max_buf)}")
        else:
            self.counter.setText("")

    def update_contraction_display(self, count, window_sec):
        self.contraction_lbl.setText(f"Contractions ({window_sec}s): {count}")


# ============================================================
# MAIN WINDOW
# ============================================================

# Colours for the action label — keyed on action string
_ACTION_COLORS = {
    "Rest":                                   "#888888",
    "Moving Forward":                         "#51cf66",
    "Moving Forward + Turning Right":         "#40c057",
    "Moving Forward + Turning Left":          "#40c057",
    "Turning Right":                          "#4dabf7",
    "Turning Left":                           "#4dabf7",
    "Dancing":                                "#ffd43b",
    "Screaming":                              "#ff6b6b",
    "Disconnected":                           "#ff6b6b",
}


class EMGRobotWindow(QWidget):
    """Unified interface: EMG visualisation + robot dog control."""

    # ── Initialisation ─────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG → Robot Dog Control")
        self.resize(1400, 780)
        self.setStyleSheet("background-color: #1a1a2e; color: white;")

        # ── EMG state ──────────────────────────────────────
        self.calibs      = [MuscleCalib(n, c) for n, c in zip(MUSCLES, COLORS)]
        self.pressed     = [False, False, False]
        self.muscle_view = 0                    # 0–2 = per-muscle; 3 = camera

        n = WINDOW_SAMPLES
        self.raw_bufs  = [np.zeros(n) for _ in range(3)]
        self.env_bufs  = [np.zeros(n) for _ in range(3)]
        self.norm_bufs = [np.zeros(n) for _ in range(3)]
        self.env_zis   = [np.zeros(len(lp_a) - 1) for _ in range(3)]

        # ── Special-command state ───────────────────────────
        self._prev_frontalis_count  = 0
        self._last_special_cmd_time = 0.0

        # Bilateral masseter tracking (both L and R above threshold at once)
        self._bilateral_contracted      = False
        self._bilateral_contraction_times = []   # onset timestamps

        # ── Robot ───────────────────────────────────────────
        self.robot           = RobotDog(ip_address=ROBOT_IP)
        self.robot_connected = False

        # ── Build UI ────────────────────────────────────────
        self._build_ui()

        # ── Main EMG timer (50 ms) ──────────────────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(50)

        # ── Camera refresh timer (~30 FPS) ──────────────────
        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self._update_camera)
        self.cam_timer.start(33)

    # ── UI construction ────────────────────────────────────
    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(10)

        # ── Left control panel ─────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        title = QLabel("EMG → ROBOT DOG")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: white; padding: 4px;")
        left.addWidget(title)

        hint = QLabel("Simulation: F = Frontalis | L = Left Masseter | R = Right Masseter")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        left.addWidget(hint)

        # Gain sliders + calibration panels per muscle
        self.sliders       = []
        self._calib_panels = []
        for i, (name, color) in enumerate(zip(MUSCLES, COLORS)):
            row = QHBoxLayout()
            lbl = QLabel(f"{name[:1]}:")
            lbl.setStyleSheet(
                f"color: {color}; font-weight: bold; min-width: 20px;")
            sl = QSlider(Qt.Horizontal)
            sl.setMinimum(0)
            sl.setMaximum(100)
            sl.setValue(50)
            val_lbl = QLabel("50")
            val_lbl.setStyleSheet("color: #aaa; min-width: 28px;")
            sl.valueChanged.connect(lambda v, vl=val_lbl: vl.setText(str(v)))
            row.addWidget(lbl)
            row.addWidget(sl)
            row.addWidget(val_lbl)
            left.addLayout(row)
            self.sliders.append(sl)

            panel = CalibPanel(self.calibs[i])
            left.addWidget(panel)
            self._calib_panels.append(panel)

        # View-switch button
        self.switch_btn = QPushButton("Switch Muscle View →")
        self.switch_btn.setStyleSheet(
            "QPushButton { background: #2b2b2b; color: #ccc; "
            "border: 1px solid #555; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #3a3a3a; }")
        self.switch_btn.clicked.connect(self._switch_view)
        left.addWidget(self.switch_btn)

        self.view_label = QLabel(f"Viewing: {MUSCLES[0]}")
        self.view_label.setStyleSheet("color: #4fc3f7; font-size: 12px;")
        left.addWidget(self.view_label)

        # Robot section
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        left.addWidget(sep)

        robot_title = QLabel("🐾 ROBOT DOG")
        robot_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ffd93d;")
        left.addWidget(robot_title)

        self.connect_btn = QPushButton("Connect to Robot")
        self.connect_btn.setStyleSheet(
            "QPushButton { background: #1a5c2a; color: white; "
            "border: 1px solid #2e8b57; border-radius: 6px; "
            "padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background: #2e8b57; }")
        self.connect_btn.clicked.connect(self._toggle_robot)
        left.addWidget(self.connect_btn)

        self.robot_status_lbl = QLabel("● Disconnected")
        self.robot_status_lbl.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        left.addWidget(self.robot_status_lbl)

        self.cmd_lbl = QLabel("Command: —")
        self.cmd_lbl.setStyleSheet(
            "color: #4fc3f7; font-size: 12px; padding: 4px;")
        left.addWidget(self.cmd_lbl)

        # Updated legend for new gesture mapping
        legend = QLabel(
            "EMG Control Mapping:\n"
            "  Frontalis active            → Move forward (speed ∝ intensity)\n"
            "  Left Masseter > Right       → Turn right  (even when stopped)\n"
            "  Right Masseter > Left       → Turn left   (even when stopped)\n"
            f"  {POWER_CONTRACTIONS}× Frontalis (10 s)       → Toggle Power\n"
            f"  Bilateral + {DANCE_FRONTALIS_CONTRACTIONS}× Frontalis (10 s) → Dance! 🐾\n"
            f"  {BUZZER_BILATERAL_CONTRACTIONS}× Bilateral Masseter (10 s) → Buzzer (2 s) 📢"
        )
        legend.setStyleSheet(
            "color: #aaa; font-size: 10px; background: #2b2b2b; "
            "padding: 6px; border-radius: 4px;")
        left.addWidget(legend)
        left.addStretch()

        main.addLayout(left, 1)

        # ── Right: stacked widget ──────────────────────────
        self.right_stack = QStackedWidget()

        # Page 0: standard 4-subplot matplotlib canvas
        self.figure = Figure(facecolor='#1a1a1a')
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(411)
        self.ax2 = self.figure.add_subplot(412)
        self.ax3 = self.figure.add_subplot(413)
        self.ax4 = self.figure.add_subplot(414)
        self.figure.tight_layout(pad=2.0)
        self.right_stack.addWidget(self.canvas)          # index 0

        # Page 1: camera view
        self.right_stack.addWidget(self._build_camera_page())  # index 1

        main.addWidget(self.right_stack, 3)

    def _build_camera_page(self):
        """
        Build the camera-view page:
          left  — 4 normalised EMG plots (frontalis, L masseter, R masseter, L-R diff)
          right — live camera feed + action-status label
        """
        page = QWidget()
        h_layout = QHBoxLayout(page)
        h_layout.setSpacing(8)

        # ── Left: mini EMG figure ──────────────────────────
        self.mini_fig = Figure(facecolor='#1a1a1a')
        self.mini_canvas = FigureCanvas(self.mini_fig)
        self.mini_canvas.setMaximumWidth(240)
        gs = self.mini_fig.add_gridspec(4, 1, hspace=0.7,
                                        left=0.18, right=0.97,
                                        top=0.96, bottom=0.05)
        self.mini_ax_f = self.mini_fig.add_subplot(gs[0])   # frontalis
        self.mini_ax_l = self.mini_fig.add_subplot(gs[1])   # left masseter
        self.mini_ax_r = self.mini_fig.add_subplot(gs[2])   # right masseter
        self.mini_ax_d = self.mini_fig.add_subplot(gs[3])   # L - R difference
        h_layout.addWidget(self.mini_canvas, 1)

        # ── Right: camera + action box ─────────────────────
        right_v = QVBoxLayout()
        right_v.setSpacing(8)

        self.camera_label = QLabel("Camera — connect robot to enable")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(520, 390)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d1a;
                border: 2px solid #2e2e5e;
                border-radius: 8px;
                color: #555;
                font-size: 14px;
            }
        """)
        right_v.addWidget(self.camera_label, 1)

        self.action_label = QLabel("Action:  Rest")
        self.action_label.setAlignment(Qt.AlignCenter)
        self.action_label.setStyleSheet(self._action_style("#888888"))
        self.action_label.setMinimumHeight(58)
        right_v.addWidget(self.action_label)

        right_widget = QWidget()
        right_widget.setLayout(right_v)
        h_layout.addWidget(right_widget, 3)

        return page

    # ── Static helper ──────────────────────────────────────
    @staticmethod
    def _action_style(color):
        return (
            f"QLabel {{"
            f"  background-color: #1e1e2e;"
            f"  color: {color};"
            f"  font-size: 22px;"
            f"  font-weight: bold;"
            f"  padding: 12px;"
            f"  border: 2px solid #333360;"
            f"  border-radius: 8px;"
            f"}}"
        )

    # ── Key simulation (keyboard → pressed[] flags) ────────
    def keyPressEvent(self, event):
        keys = [Qt.Key_F, Qt.Key_L, Qt.Key_R]
        for i, k in enumerate(keys):
            if event.key() == k:
                self.pressed[i] = True

    def keyReleaseEvent(self, event):
        keys = [Qt.Key_F, Qt.Key_L, Qt.Key_R]
        for i, k in enumerate(keys):
            if event.key() == k:
                self.pressed[i] = False

    # ── View switching (0–2 = per-muscle plot; 3 = camera) ─
    def _switch_view(self):
        self.muscle_view = (self.muscle_view + 1) % 4
        if self.muscle_view == 3:
            self.view_label.setText("Viewing: Camera + EMG")
            self.switch_btn.setText("← Back to Muscle View")
            self.right_stack.setCurrentIndex(1)
        else:
            self.view_label.setText(f"Viewing: {MUSCLES[self.muscle_view]}")
            self.switch_btn.setText("Switch Muscle View →")
            self.right_stack.setCurrentIndex(0)

    # ── Robot connect / disconnect ─────────────────────────
    def _toggle_robot(self):
        if not self.robot_connected:
            ok = self.robot.connect()
            if ok:
                self.robot_connected = True
                self.connect_btn.setText("Disconnect Robot")
                self.robot_status_lbl.setText("● Connected")
                self.robot_status_lbl.setStyleSheet(
                    "color: #51cf66; font-weight: bold;")
        else:
            self.robot.disconnect()
            self.robot_connected = False
            self.connect_btn.setText("Connect to Robot")
            self.robot_status_lbl.setText("● Disconnected")
            self.robot_status_lbl.setStyleSheet(
                "color: #ff6b6b; font-weight: bold;")

    # ── Main update loop (50 ms timer) ─────────────────────
    def _update(self):
        t_now     = time.time()
        raws      = generate_signals(self.pressed, [s.value() for s in self.sliders])
        norm_vals = []

        for i in range(3):
            env, raw, self.env_zis[i] = process_chunk(raws[i], self.env_zis[i])
            self.calibs[i].feed(env)
            norm = self.calibs[i].normalize(env)
            self.calibs[i].window_sec = 10.0
            self.calibs[i].update_contraction(float(norm[-1]), t_now)

            self.raw_bufs[i]  = np.roll(self.raw_bufs[i],  -CHUNK)
            self.raw_bufs[i][-CHUNK:]  = raws[i]
            self.env_bufs[i]  = np.roll(self.env_bufs[i],  -CHUNK)
            self.env_bufs[i][-CHUNK:]  = env
            self.norm_bufs[i] = np.roll(self.norm_bufs[i], -CHUNK)
            self.norm_bufs[i][-CHUNK:] = norm

            norm_vals.append(float(norm[-1]))

        # ── Bilateral masseter contraction tracking ────────
        # A "bilateral event" = onset of both L and R above their thresholds
        both_active = (
            norm_vals[1] > self.calibs[1].threshold and
            norm_vals[2] > self.calibs[2].threshold
        )
        if both_active and not self._bilateral_contracted:
            self._bilateral_contracted = True
            self._bilateral_contraction_times.append(t_now)
        elif not both_active:
            self._bilateral_contracted = False

        # Prune events older than 10 s
        self._bilateral_contraction_times = [
            t for t in self._bilateral_contraction_times
            if t >= t_now - 10.0
        ]
        bilateral_count = len(self._bilateral_contraction_times)

        # ── Update calibration panels ──────────────────────
        for i, panel in enumerate(self._calib_panels):
            panel.update_counter()
            panel.status.setText(self.calibs[i].status_text())
            panel.update_contraction_display(
                self.calibs[i].count_contractions(t_now), 10)

        # ── EMG → robot (updated signature) ───────────────
        self._prev_frontalis_count, self._last_special_cmd_time = emg_to_robot(
            self.robot, self.calibs, norm_vals,
            bilateral_count, self._bilateral_contraction_times,
            t_now,
            self._prev_frontalis_count, self._last_special_cmd_time,
            self.cmd_lbl,
        )

        # ── Render ─────────────────────────────────────────
        if self.muscle_view == 3:
            self._draw_mini_plot()
            self._update_action_label()
        else:
            self._draw_plot()

    # ── Camera frame refresh (~30 FPS, only when in view 3) ─
    def _update_camera(self):
        if self.muscle_view != 3:
            return
        if not self.robot_connected:
            return
        if not _CV2_OK:
            self.camera_label.setText("cv2 not installed — pip install opencv-python")
            return
        try:
            client_img = getattr(self.robot.client, 'image', None)
            if client_img is not None and len(client_img) > 0:
                frame     = client_img.copy()
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w      = rgb.shape[:2]
                q_img     = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap    = QPixmap.fromImage(q_img)
                scaled    = pixmap.scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.camera_label.setPixmap(scaled)
        except Exception:
            pass   # silently skip bad frames

    # ── Mini EMG plots (view 3 only) ───────────────────────
    def _draw_mini_plot(self):
        t = np.linspace(-WINDOW_SEC, 0, WINDOW_SAMPLES)

        mini_axes = [self.mini_ax_f, self.mini_ax_l, self.mini_ax_r, self.mini_ax_d]
        for ax in mini_axes:
            ax.clear()
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='#888888', labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor('#333333')

        # Frontalis normalised
        self.mini_ax_f.plot(t, self.norm_bufs[0], color=COLORS[0], linewidth=0.9)
        self.mini_ax_f.axhline(self.calibs[0].threshold,
                                color='#00ff88', linewidth=1.2, linestyle='--')
        self.mini_ax_f.set_ylim(-0.05, 1.05)
        self.mini_ax_f.set_title("Frontalis", color=COLORS[0], fontsize=8, pad=2)
        self.mini_ax_f.set_ylabel("Norm", color='#888', fontsize=7)
        self.mini_ax_f.set_xticklabels([])

        # Left masseter normalised
        self.mini_ax_l.plot(t, self.norm_bufs[1], color=COLORS[1], linewidth=0.9)
        self.mini_ax_l.axhline(self.calibs[1].threshold,
                                color='#00ff88', linewidth=1.2, linestyle='--')
        self.mini_ax_l.set_ylim(-0.05, 1.05)
        self.mini_ax_l.set_title("Left Masseter", color=COLORS[1], fontsize=8, pad=2)
        self.mini_ax_l.set_ylabel("Norm", color='#888', fontsize=7)
        self.mini_ax_l.set_xticklabels([])

        # Right masseter normalised
        self.mini_ax_r.plot(t, self.norm_bufs[2], color=COLORS[2], linewidth=0.9)
        self.mini_ax_r.axhline(self.calibs[2].threshold,
                                color='#00ff88', linewidth=1.2, linestyle='--')
        self.mini_ax_r.set_ylim(-0.05, 1.05)
        self.mini_ax_r.set_title("Right Masseter", color=COLORS[2], fontsize=8, pad=2)
        self.mini_ax_r.set_ylabel("Norm", color='#888', fontsize=7)
        self.mini_ax_r.set_xticklabels([])

        # L - R difference
        diff = self.norm_bufs[1] - self.norm_bufs[2]
        self.mini_ax_d.plot(t, diff, color='#d896ff', linewidth=0.9)
        self.mini_ax_d.axhline( MASSETER_DIFF_THRESHOLD,
                                 color='#ff5555', linewidth=1, linestyle='--')
        self.mini_ax_d.axhline(-MASSETER_DIFF_THRESHOLD,
                                 color='#55aaff', linewidth=1, linestyle='--')
        self.mini_ax_d.axhline(0, color='#555555', linewidth=0.6)
        self.mini_ax_d.set_ylim(-1.05, 1.05)
        self.mini_ax_d.set_title("L − R Diff", color='#d896ff', fontsize=8, pad=2)
        self.mini_ax_d.set_ylabel("Δ", color='#888', fontsize=7)
        self.mini_ax_d.set_xlabel("Time (s)", color='#888', fontsize=7)

        self.mini_canvas.draw()

    # ── Action label helpers ───────────────────────────────
    def _get_current_action(self):
        """Return an English string describing the robot's current action."""
        if not self.robot_connected:
            return "Disconnected"
        if self.robot.is_dancing:
            return "Dancing"
        if self.robot._buzzer_active:
            return "Screaming"

        frontalis_on = self.robot.emg_frontalis_intensity > self.robot.frontalis_threshold
        diff         = self.robot.emg_masseter_diff
        turn_r       = diff >  self.robot.masseter_threshold
        turn_l       = diff < -self.robot.masseter_threshold

        if frontalis_on and turn_r:
            return "Moving Forward + Turning Right"
        if frontalis_on and turn_l:
            return "Moving Forward + Turning Left"
        if frontalis_on:
            return "Moving Forward"
        if turn_r:
            return "Turning Right"
        if turn_l:
            return "Turning Left"
        return "Rest"

    def _update_action_label(self):
        action = self._get_current_action()
        color  = _ACTION_COLORS.get(action, "#00ff88")
        self.action_label.setText(f"Action:  {action}")
        self.action_label.setStyleSheet(self._action_style(color))

    # ── Standard 4-subplot plot (views 0–2) ───────────────
    def _draw_plot(self):
        i    = self.muscle_view
        c    = COLORS[i]
        name = MUSCLES[i]
        cal  = self.calibs[i]
        t    = np.linspace(-WINDOW_SEC, 0, WINDOW_SAMPLES)

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.clear()
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='#888888', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

        self.ax1.plot(t, self.raw_bufs[i], color=c, linewidth=0.7)
        self.ax1.set_title(f"RAW — {name}", color='white', fontsize=10)
        self.ax1.set_ylabel("V", color='#888')

        self.ax2.plot(t, self.env_bufs[i], color=c, linewidth=0.9)
        self.ax2.set_title("ENVELOPE", color='white', fontsize=10)
        self.ax2.set_ylabel("V", color='#888')
        if cal.calibrated:
            self.ax2.axhline(cal.thresh_lo, color='#ffcc02',
                             linewidth=1, linestyle='--')
            self.ax2.axhline(cal.thresh_hi, color='#ff5555',
                             linewidth=1, linestyle='--')

        self.ax3.plot(t, self.norm_bufs[i], color=c, linewidth=0.9)
        self.ax3.axhline(
            FRONTALIS_NORM_THRESHOLD if i == 0 else cal.threshold,
            color='#00ff88', linewidth=1.5, linestyle='-',
            label=f'Threshold: {cal.threshold:.2f}',
        )
        self.ax3.set_ylim(-0.05, 1.05)
        self.ax3.set_title("NORMALIZED (0–1)", color='white', fontsize=10)
        self.ax3.legend(loc='upper right', facecolor='#2b2b2b',
                        labelcolor='white', fontsize=8)

        diff = self.norm_bufs[1] - self.norm_bufs[2]
        self.ax4.plot(t, diff, color='#d896ff', linewidth=1.0)
        self.ax4.axhline( MASSETER_DIFF_THRESHOLD,
                           color='#ff5555', linewidth=1, linestyle='--')
        self.ax4.axhline(-MASSETER_DIFF_THRESHOLD,
                           color='#55aaff', linewidth=1, linestyle='--')
        self.ax4.axhline(0, color='#888888', linewidth=0.8)
        self.ax4.set_ylim(-1.05, 1.05)
        self.ax4.set_title("L − R DIFFERENCE", color='white', fontsize=10)
        self.ax4.set_xlabel("Time (s)", color='#888')

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()

    # ── Window close ──────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        self.cam_timer.stop()
        if self.robot_connected:
            self.robot.disconnect()
        event.accept()

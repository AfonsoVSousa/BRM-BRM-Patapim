# -*- coding: utf-8 -*-
"""
robot_controller.py — RobotDog class and EMG → command mapping.

Changes vs. original:
  • Turning is now independent of frontalis (robot can turn even when stopped).
  • New emg_buzzer_trigger flag: activates buzzer for 2 s.
  • Video-receiving thread started on connect() so camera feed is available.
  • emg_to_robot() updated: new bilateral gesture parameters, always writes
    emg_masseter_diff regardless of frontalis state, English command labels.
"""

import time
import threading

from Client  import Client
from Command import COMMAND as cmd

from config import (
    FRONTALIS_NORM_THRESHOLD, MASSETER_DIFF_THRESHOLD,
    POWER_CONTRACTIONS,
    DANCE_BILATERAL_CONTRACTIONS, DANCE_FRONTALIS_CONTRACTIONS,
    BUZZER_BILATERAL_CONTRACTIONS,
    ROBOT_IP,
)


# ============================================================
# ROBOT DOG
# ============================================================

class RobotDog:
    """Wraps the Client / COMMAND API and exposes EMG control variables."""

    def __init__(self, ip_address=ROBOT_IP):
        self.IP     = ip_address
        self.client = Client()
        self.client.move_speed = "5"
        self.is_running = False

        # ── EMG control variables (written by UI, read by control loop) ──
        self.emg_frontalis_intensity = 0.0   # 0–100 (speed proxy)
        self.emg_masseter_diff       = 0.0   # <0 → left turn, >0 → right turn
        self.emg_power_toggle        = False
        self.emg_dance_trigger       = False
        self.emg_buzzer_trigger      = False  # NEW: 2-s buzzer activation

        # ── Thresholds ───────────────────────────────────────
        self.frontalis_threshold     = 20.0
        self.masseter_threshold      = 5.0
        self.timeout_duration        = 2.0
        self.last_valid_command_time = time.time()

        # ── Status flags ─────────────────────────────────────
        self.is_dancing    = False
        self._buzzer_active = False   # True while 2-s beep is running

    # ── Connection ─────────────────────────────────────────
    def connect(self):
        print("Connecting to robot...")
        self.client.turn_on_client(self.IP)
        try:
            self.client.client_socket1.connect((self.IP, 5001))
            self.client.tcp_flag = True
            print("Connected!")
            self.is_running = True

            # Start video-receiving thread so camera feed is available
            self._video_thread = threading.Thread(
                target=self.client.receiving_video,
                args=(self.IP,),
                daemon=True,
            )
            self._video_thread.start()

            # Start control loop thread
            self.control_thread = threading.Thread(
                target=self._control_loop, daemon=True)
            self.control_thread.start()
            return True
        except Exception as e:
            print("Connection failed:", e)
            self.client.tcp_flag = False
            return False

    def disconnect(self):
        self.is_running = False
        self._stop_movement()
        self._set_led(0)
        self.client.tcp_flag = False
        self.client.turn_off_client()
        print("Robot disconnected.")

    # ── Control loop (background thread) ───────────────────
    def _control_loop(self):
        robot_active = True
        self._set_led(1)

        while self.is_running:
            try:
                now = time.time()

                # ── Priority 1: dance ──────────────────────
                if self.emg_dance_trigger:
                    self.emg_dance_trigger = False
                    threading.Thread(target=self._dance, daemon=True).start()
                    time.sleep(0.05)
                    continue

                if self.is_dancing:
                    time.sleep(0.05)
                    continue

                # ── Priority 2: buzzer ─────────────────────
                if self.emg_buzzer_trigger:
                    self.emg_buzzer_trigger = False
                    threading.Thread(target=self._buzzer_alert, daemon=True).start()
                    time.sleep(0.05)
                    continue

                # ── Priority 3: power toggle ───────────────
                if self.emg_power_toggle:
                    robot_active = not robot_active
                    print(f"[POWER] {'Active' if robot_active else 'Inactive'}")
                    self._set_led(1 if robot_active else 0)
                    if not robot_active:
                        self._stop_movement()
                    self.emg_power_toggle = False
                    time.sleep(0.6)
                    continue

                # ── Priority 4: movement ───────────────────
                if robot_active:
                    frontalis_active = (
                        self.emg_frontalis_intensity > self.frontalis_threshold
                    )
                    turn_right = self.emg_masseter_diff >  self.masseter_threshold
                    turn_left  = self.emg_masseter_diff < -self.masseter_threshold

                    if frontalis_active:
                        # Speed proportional to frontalis intensity (range 2–10)
                        raw_spd = 2 + (self.emg_frontalis_intensity / 100.0) * 8
                        spd = str(int(max(2, min(10, raw_spd))))
                        self.client.move_speed = spd

                        if turn_right:
                            self.client.send_data(
                                cmd.CMD_TURN_RIGHT + "#" + spd + '\n')
                        elif turn_left:
                            self.client.send_data(
                                cmd.CMD_TURN_LEFT + "#" + spd + '\n')
                        else:
                            self.client.send_data(
                                cmd.CMD_MOVE_FORWARD + "#" + spd + '\n')

                        self.last_valid_command_time = now

                    elif turn_right:
                        # Frontalis off but right masseter dominant → turn in place
                        self.client.send_data(cmd.CMD_TURN_RIGHT + "#4\n")
                        self.last_valid_command_time = now

                    elif turn_left:
                        # Frontalis off but left masseter dominant → turn in place
                        self.client.send_data(cmd.CMD_TURN_LEFT + "#4\n")
                        self.last_valid_command_time = now

                    else:
                        self._stop_movement()

                    # Safety stop on communication timeout
                    if (now - self.last_valid_command_time) > self.timeout_duration:
                        self._stop_movement()

                time.sleep(0.05)

            except Exception as e:
                print("Control loop error:", e)

    # ── Low-level helpers ──────────────────────────────────
    def _stop_movement(self):
        spd = self.client.move_speed or "5"
        self.client.send_data(cmd.CMD_MOVE_STOP + "#" + spd + '\n')

    def _set_led(self, mode):
        self.client.send_data(cmd.CMD_LED_MOD + '#' + str(mode) + '\n')

    def _beep(self, on_t, off_t=0.05):
        self.client.send_data(cmd.CMD_BUZZER + '#1\n')
        time.sleep(on_t)
        self.client.send_data(cmd.CMD_BUZZER + '#0\n')
        time.sleep(off_t)

    def _buzzer_alert(self):
        """Activate buzzer continuously for 2 seconds."""
        self._buzzer_active = True
        try:
            self.client.send_data(cmd.CMD_BUZZER + '#1\n')
            time.sleep(2.0)
            self.client.send_data(cmd.CMD_BUZZER + '#0\n')
        except Exception as e:
            print("[BUZZER] Error:", e)
        finally:
            self._buzzer_active = False

    def _dance(self):
        self.is_dancing = True
        spd = "4"

        def s(c):
            self.client.send_data(c)

        def led(r, g, b):
            s(cmd.CMD_LED_MOD + '#1\n')
            s(cmd.CMD_LED + f'#255#{r}#{g}#{b}\n')

        try:
            for r, g, b in [(255,0,0),(0,255,0),(0,0,255),(255,165,0),(128,0,255)]:
                led(r, g, b)
                time.sleep(0.3)
            led(0, 200, 255)
            for _ in range(4):
                s(cmd.CMD_HEIGHT + '#-15\n')
                self._beep(0.06, 0)
                time.sleep(0.19)
                s(cmd.CMD_HEIGHT + '#15\n')
                self._beep(0.06, 0)
                time.sleep(0.19)
            s(cmd.CMD_HEIGHT + '#0\n')
            led(255, 100, 0)
            for _ in range(4):
                s(cmd.CMD_HEAD + '#60\n')
                time.sleep(0.2)
                s(cmd.CMD_HEAD + '#120\n')
                time.sleep(0.2)
            s(cmd.CMD_HEAD + '#90\n')
            led(0, 255, 100)
            for _ in range(2):
                s(cmd.CMD_MOVE_LEFT  + '#' + spd + '\n')
                time.sleep(0.4)
                s(cmd.CMD_MOVE_STOP  + '#' + spd + '\n')
                time.sleep(0.1)
                s(cmd.CMD_MOVE_RIGHT + '#' + spd + '\n')
                time.sleep(0.4)
                s(cmd.CMD_MOVE_STOP  + '#' + spd + '\n')
                time.sleep(0.1)
            led(255, 0, 200)
            s(cmd.CMD_TURN_LEFT  + '#' + spd + '\n')
            time.sleep(1.0)
            s(cmd.CMD_TURN_RIGHT + '#' + spd + '\n')
            time.sleep(1.0)
            s(cmd.CMD_MOVE_STOP  + '#' + spd + '\n')
            for r, g, b in [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,255,255)]:
                led(r, g, b)
                time.sleep(0.12)
        except Exception as e:
            print("[DANCE] Error:", e)
        finally:
            self._stop_movement()
            s(cmd.CMD_HEIGHT  + '#0\n')
            s(cmd.CMD_HORIZON + '#0\n')
            s(cmd.CMD_HEAD    + '#90\n')
            self._set_led(1)
            self.is_dancing = False
            print("[DANCE] Done!")


# ============================================================
# EMG → ROBOT COMMAND MAPPING
# ============================================================

def emg_to_robot(robot, calibs, norm_vals,
                 bilateral_count, bilateral_times,
                 t_now, prev_frontalis_count, last_special_cmd_time,
                 cmd_lbl):
    """
    Map normalised EMG values → robot flags and UI label.

    Parameters
    ----------
    robot               : RobotDog
    calibs              : list[MuscleCalib]  — [frontalis, left, right]
    norm_vals           : list[float]        — current normalised value per muscle
    bilateral_count     : int                — simultaneous L+R events in last 10 s
    bilateral_times     : list               — mutable; cleared here when gesture fires
    t_now               : float              — current timestamp
    prev_frontalis_count: int
    last_special_cmd_time: float
    cmd_lbl             : QLabel

    Returns
    -------
    (prev_frontalis_count, last_special_cmd_time)
    """

    frontalis = norm_vals[0]
    left      = norm_vals[1]
    right     = norm_vals[2]

    # Masseter difference drives direction; always written regardless of frontalis
    D = left - right
    robot.emg_masseter_diff = D * 100.0

    # ── Special gesture detection ──────────────────────────
    front_count     = calibs[0].count_contractions(t_now)
    time_since_last = t_now - last_special_cmd_time

    if time_since_last > 1.0:

        # Priority 1 — Buzzer: 2× bilateral masseter clenches
        if bilateral_count >= BUZZER_BILATERAL_CONTRACTIONS:
            cmd_lbl.setText("Command: 📢 Screaming")
            robot.emg_buzzer_trigger = True
            last_special_cmd_time    = t_now
            bilateral_times.clear()

        # Priority 2 — Dance: ≥1 bilateral event AND ≥2 frontalis contractions
        elif (bilateral_count >= DANCE_BILATERAL_CONTRACTIONS
              and front_count >= DANCE_FRONTALIS_CONTRACTIONS
              and not robot.is_dancing):
            cmd_lbl.setText("Command: 🐾 Dancing!")
            robot.emg_dance_trigger  = True
            last_special_cmd_time    = t_now
            calibs[0].contraction_times.clear()
            bilateral_times.clear()

        # Priority 3 — Power toggle: 3× frontalis alone
        elif front_count >= POWER_CONTRACTIONS:
            cmd_lbl.setText("Command: ⚡ Power Toggle")
            robot.emg_power_toggle  = True
            last_special_cmd_time   = t_now
            calibs[0].contraction_times.clear()

    prev_frontalis_count = front_count

    # ── Movement / direction label ─────────────────────────
    frontalis_active = frontalis >  FRONTALIS_NORM_THRESHOLD
    turn_right       = D         >  MASSETER_DIFF_THRESHOLD
    turn_left        = D         < -MASSETER_DIFF_THRESHOLD

    if frontalis_active:
        robot.emg_frontalis_intensity = frontalis * 100.0
        if turn_right:
            cmd_lbl.setText("Command: Moving Forward + Turning Right →")
        elif turn_left:
            cmd_lbl.setText("Command: Moving Forward + Turning Left ←")
        else:
            cmd_lbl.setText("Command: Moving Forward ↑")
    else:
        robot.emg_frontalis_intensity = 0.0
        if turn_right:
            cmd_lbl.setText("Command: Turning Right →")
        elif turn_left:
            cmd_lbl.setText("Command: Turning Left ←")
        else:
            cmd_lbl.setText("Command: Rest —")

    return prev_frontalis_count, last_special_cmd_time

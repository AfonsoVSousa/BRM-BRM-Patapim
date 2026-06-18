# -*- coding: utf-8 -*-
"""
emg_core.py — MuscleCalib class + signal generation + processing.
No Qt dependency — pure numpy/scipy.
"""

import numpy as np
from scipy.signal import lfilter

from config import (
    ADAPTIVE_LMS_MU,
    CHUNK,
    FS,
    NOTCH_FREQ,
    bp_b,
    bp_a,
    lp_b,
    lp_a,
)


# ============================================================
# MUSCLE CALIBRATION
# ============================================================

class MuscleCalib:
    """Calibration state and thresholds for one muscle channel."""

    def __init__(self, name, color):
        self.name  = name
        self.color = color

        # calibration buffers
        self.rest_buf = []
        self.max_buf  = []
        self.cal_rest = False
        self.cal_max  = False

        # computed levels
        self.rest_mean = 0.0
        self.max_mean  = 1.0
        self.thresh_lo = 0.1
        self.thresh_hi = 0.9
        self.calibrated = False

        # contraction detection
        self.threshold         = 0.5
        self.contracted        = False
        self.contraction_start = None
        self.contraction_times = []
        self.window_sec        = 10.0

    # ── Calibration control ────────────────────────────────
    def start_rest(self):
        self.cal_rest = True
        self.rest_buf = []

    def stop_rest(self):
        self.cal_rest = False
        if len(self.rest_buf) > 10:
            self.rest_mean = float(np.mean(self.rest_buf))
        self._recompute()

    def start_max(self):
        self.cal_max = True
        self.max_buf = []

    def stop_max(self):
        self.cal_max = False
        if len(self.max_buf) > 10:
            self.max_mean = float(np.mean(self.max_buf))
        self._recompute()

    def feed(self, env_chunk):
        """Feed envelope samples into active calibration buffer."""
        if self.cal_rest:
            self.rest_buf.extend(env_chunk.tolist())
        if self.cal_max:
            self.max_buf.extend(env_chunk.tolist())

    def _recompute(self):
        rang = self.max_mean - self.rest_mean
        if rang > 1e-9:
            self.thresh_lo  = self.rest_mean + 0.10 * rang
            self.thresh_hi  = self.rest_mean + 0.90 * rang
            self.calibrated = True

    def normalize(self, env):
        denom = (self.max_mean - self.rest_mean + 1e-9)
        return np.clip((env - self.rest_mean) / denom, 0, 1)

    # ── Contraction detection ──────────────────────────────
    def update_contraction(self, norm_value, t):
        above = norm_value > self.threshold
        if above and not self.contracted:
            self.contracted        = True
            self.contraction_start = t
        elif not above and self.contracted:
            self.contracted = False
            if self.contraction_start is not None:
                self.contraction_times.append(self.contraction_start)
            self.contraction_start = None

    def count_contractions(self, t):
        cutoff = t - self.window_sec
        self.contraction_times = [s for s in self.contraction_times if s >= cutoff]
        count = len(self.contraction_times)
        if self.contracted and self.contraction_start is not None \
                and self.contraction_start >= cutoff:
            count += 1
        return count

    def status_text(self):
        if not self.calibrated:
            return "Not calibrated"
        return (f"REST {self.rest_mean:.3f} | MAX {self.max_mean:.3f} | "
                f"LO {self.thresh_lo:.3f} | HI {self.thresh_hi:.3f}")


class Adaptive50HzCanceller:
    """Two-tap LMS canceller for a narrow 50 Hz interference line."""

    def __init__(self, fs=FS, freq=NOTCH_FREQ, mu=ADAPTIVE_LMS_MU):
        self.fs = fs
        self.freq = freq
        self.mu = mu
        self.phase = 0.0
        self.weights = np.zeros(2, dtype=np.float64)

    def process(self, samples):
        omega = 2.0 * np.pi * self.freq / self.fs
        output = np.empty_like(samples, dtype=np.float64)

        for index, sample in enumerate(samples):
            ref = np.array([
                np.sin(self.phase),
                np.cos(self.phase),
            ], dtype=np.float64)
            estimate = float(np.dot(self.weights, ref))
            error = float(sample - estimate)
            self.weights += self.mu * error * ref
            output[index] = error
            self.phase += omega
            if self.phase >= 2.0 * np.pi:
                self.phase -= 2.0 * np.pi

        return output


# ============================================================
# SIGNAL GENERATION (keyboard simulation)
# ============================================================

def generate_signals(pressed, sliders):
    """
    Simulate 3-channel EMG from keyboard state.
    pressed : list[bool]   — [F, L, R] key state
    sliders : list[int]    — slider values 0-100 per muscle
    Returns list of 3 numpy arrays of length CHUNK.
    """
    base  = lfilter([1], [1, -0.9], np.random.normal(0, 1.0, CHUNK))
    out   = []
    for i in range(3):
        if pressed[i]:
            out.append((sliders[i] / 100.0) * base)
        else:
            out.append(np.random.normal(0, 0.02, CHUNK))
    return out


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def process_chunk(raw, bp_zi, notch_state, env_zi):
    """
    Bandpass filter → adaptive 50 Hz canceller → full-wave rectify → envelope.
    Returns (envelope, raw_filtered, new_bp_zi, notch_state, new_env_zi).
    """
    filt, bp_zi = lfilter(bp_b, bp_a, raw, zi=bp_zi)
    filt = notch_state.process(filt)
    rect = np.abs(filt)
    env, env_zi = lfilter(lp_b, lp_a, rect, zi=env_zi)
    return env, filt, bp_zi, notch_state, env_zi

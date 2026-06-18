# -*- coding: utf-8 -*-
"""
config.py — All constants and filter coefficients for EMG → Robot Dog.
"""

from scipy.signal import butter

# ── EMG Acquisition ────────────────────────────────────────
FS             = 2000
WINDOW_SEC     = 5
WINDOW_SAMPLES = FS * WINDOW_SEC
CHUNK          = 100    # 50 ms

# ── Filter Parameters ──────────────────────────────────────
LOWCUT          = 20
HIGHCUT         = 450
NOTCH_FREQ      = 50
ADAPTIVE_LMS_MU = 0.002
ENVELOPE_CUTOFF = 2

# ── Muscles ────────────────────────────────────────────────
MUSCLES = ["Frontalis", "Left Masseter", "Right Masseter"]
COLORS  = ["#4fc3f7", "#81c784", "#ff8a65"]

# ── EMG Decision Thresholds ────────────────────────────────
FRONTALIS_NORM_THRESHOLD = 0.5   # above this → contracting
MASSETER_DIFF_THRESHOLD  = 0.2   # minimum |L − R| for direction decision

# ── Special Command Counts (within 10 s sliding window) ────
# Power toggle: 3 frontalis contractions alone
POWER_CONTRACTIONS = 3

# Dance: ≥1 bilateral masseter event AND ≥2 frontalis contractions
DANCE_BILATERAL_CONTRACTIONS = 1
DANCE_FRONTALIS_CONTRACTIONS = 2

# Buzzer (2 s): 2 simultaneous bilateral masseter events
BUZZER_BILATERAL_CONTRACTIONS = 2

# ── Robot ──────────────────────────────────────────────────
ROBOT_IP = "10.107.71.114"

# ── Filter Coefficients (computed once at import) ──────────
def _bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def _lowpass(cutoff, fs, order=2):
    nyq = fs / 2
    return butter(order, cutoff / nyq, btype='low')

bp_b, bp_a = _bandpass(LOWCUT, HIGHCUT, FS)
lp_b, lp_a = _lowpass(ENVELOPE_CUTOFF, FS)

# ── DAQ Channels ───────────────────────────────────────────────────────
# Verify device name in NI MAX before running (may be Dev1, Dev2, etc.)
DAQ_AO_CHANNEL  = b"Dev1/ao0"   # AO output — frontalis signal generator
DAQ_AI_CHANNEL  = b"Dev1/ai1"   # AI input  — signal after analog chain
DAQ_V_MIN       = -1.0           # V — must cover INA126 output swing
DAQ_V_MAX       =  1.0           # V
DAQ_AMPLITUDE   = 0.0008            # V — AO peak amplitude when F is held
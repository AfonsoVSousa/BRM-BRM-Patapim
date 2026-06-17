# -*- coding: utf-8 -*-
import ctypes
import threading
import numpy as np
from collections import deque
from scipy.signal import lfilter

from config import CHUNK, FS, DAQ_AMPLITUDE

# ── Load DLL ───────────────────────────────────────────────────────────
try:
    _dll           = ctypes.cdll.LoadLibrary(r"C:\Windows\System32\nicaiu.dll")
    _DAQ_AVAILABLE = True

    _dll.DAQmxCreateTask.argtypes        = [ctypes.c_char_p,
                                             ctypes.POINTER(ctypes.c_void_p)]
    _dll.DAQmxCreateTask.restype         = ctypes.c_int32

    _dll.DAQmxCreateAOVoltageChan.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                               ctypes.c_char_p, ctypes.c_double,
                                               ctypes.c_double, ctypes.c_int32,
                                               ctypes.c_char_p]
    _dll.DAQmxCreateAOVoltageChan.restype  = ctypes.c_int32

    _dll.DAQmxCreateAIVoltageChan.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                               ctypes.c_char_p, ctypes.c_int32,
                                               ctypes.c_double, ctypes.c_double,
                                               ctypes.c_int32, ctypes.c_char_p]
    _dll.DAQmxCreateAIVoltageChan.restype  = ctypes.c_int32

    _dll.DAQmxCfgSampClkTiming.argtypes  = [ctypes.c_void_p, ctypes.c_char_p,
                                             ctypes.c_double, ctypes.c_int32,
                                             ctypes.c_int32, ctypes.c_uint64]
    _dll.DAQmxCfgSampClkTiming.restype   = ctypes.c_int32

    _dll.DAQmxWriteAnalogF64.argtypes    = [ctypes.c_void_p, ctypes.c_int32,
                                             ctypes.c_uint32, ctypes.c_double,
                                             ctypes.c_int32,
                                             ctypes.POINTER(ctypes.c_double),
                                             ctypes.POINTER(ctypes.c_int32),
                                             ctypes.c_void_p]
    _dll.DAQmxWriteAnalogF64.restype     = ctypes.c_int32

    _dll.DAQmxReadAnalogF64.argtypes     = [ctypes.c_void_p, ctypes.c_int32,
                                             ctypes.c_double, ctypes.c_int32,
                                             ctypes.POINTER(ctypes.c_double),
                                             ctypes.c_uint32,
                                             ctypes.POINTER(ctypes.c_int32),
                                             ctypes.c_void_p]
    _dll.DAQmxReadAnalogF64.restype      = ctypes.c_int32

    _dll.DAQmxStartTask.argtypes         = [ctypes.c_void_p]
    _dll.DAQmxStartTask.restype          = ctypes.c_int32

    _dll.DAQmxStopTask.argtypes          = [ctypes.c_void_p]
    _dll.DAQmxStopTask.restype           = ctypes.c_int32

    _dll.DAQmxClearTask.argtypes         = [ctypes.c_void_p]
    _dll.DAQmxClearTask.restype          = ctypes.c_int32

except Exception as _err:
    _dll           = None
    _DAQ_AVAILABLE = False
    print(f"[DAQ] nicaiu.dll not available: {_err}")

# ── DAQmx constants ────────────────────────────────────────────────────
_VAL_VOLTS         = 10348
_VAL_DIFF          = 10106
_VAL_CONT_SAMPS    = 10123
_VAL_RISING        = 10280
_VAL_GROUP_BY_CHAN = 0


class FrontalisDAQ:

    def __init__(self,
                 ao_channel = b"Dev1/ao0",
                 ai_channel = b"Dev1/ai1",
                 v_min      = -1.0,
                 v_max      =  1.0,
                 amplitude  = DAQ_AMPLITUDE):

        self.ao_channel = ao_channel
        self.ai_channel = ai_channel
        self.v_min      = v_min
        self.v_max      = v_max
        self.amplitude  = amplitude

        self._ao_task   = None
        self._ai_task   = None
        self._running   = False

        self._active    = False
        self._ar_zi     = np.zeros(1)

        self._lock      = threading.Lock()
        self._buffer    = deque(maxlen=CHUNK * 10)

        self.connected  = False
        self.error_msg  = ""

    # ── Public API ─────────────────────────────────────────────────────

    def set_active(self, active: bool):
        self._active = active

    def read_chunk(self):
        with self._lock:
            if len(self._buffer) >= CHUNK:
                return np.array(
                    [self._buffer.popleft() for _ in range(CHUNK)],
                    dtype=np.float64,
                )
        return None

    def start(self):
        if not _DAQ_AVAILABLE:
            self.error_msg = "nicaiu.dll not found"
            return False

        result = [None]
        done   = threading.Event()

        def _worker():
            try:
                result[0] = self._setup_tasks()
            except Exception as e:
                self.error_msg = str(e)
                result[0] = False
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        done.wait(timeout=10.0)

        if result[0] is None:
            self.error_msg = "DAQ setup timed out"
            return False
        return result[0]

    def stop(self):
        self._running  = False
        self.connected = False
        for task in (self._ao_task, self._ai_task):
            if task is not None:
                try:
                    _dll.DAQmxStopTask(task)
                    _dll.DAQmxClearTask(task)
                except Exception:
                    pass
        self._ao_task = None
        self._ai_task = None

    # ── Setup ──────────────────────────────────────────────────────────

    def _setup_tasks(self):
        print("[DAQ] _setup_tasks iniciado")

        # ── AO ────────────────────────────────────────────────────────
        self._ao_task = ctypes.c_void_p(0)
        print("[DAQ] antes CreateTask AO")
        err = _dll.DAQmxCreateTask(b"", ctypes.byref(self._ao_task))
        print(f"[DAQ] CreateTask AO: err={err}, handle={self._ao_task.value}")
        if err != 0:
            self.error_msg = f"AO: DAQmxCreateTask failed ({err})"
            return False

        err = _dll.DAQmxCreateAOVoltageChan(
            self._ao_task, self.ao_channel, b"",
            ctypes.c_double(-10.0),
            ctypes.c_double(10.0),
            _VAL_VOLTS, None,
        )
        print(f"[DAQ] CreateAOVoltageChan: err={err}")
        if err != 0:
            self.error_msg = f"AO: CreateAOVoltageChan failed ({err})"
            return False

        err = _dll.DAQmxCfgSampClkTiming(
            self._ao_task, b"",
            ctypes.c_double(float(FS)),
            _VAL_RISING, _VAL_CONT_SAMPS,
            ctypes.c_uint64(CHUNK * 50),
        )
        print(f"[DAQ] CfgSampClkTiming AO: err={err}")
        if err != 0:
            self.error_msg = f"AO: CfgSampClkTiming failed ({err})"
            return False

        prefill  = CHUNK * 4
        pre_data = np.zeros(prefill, dtype=np.float64)
        pre_buf  = pre_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        written  = ctypes.c_int32(0)
        err = _dll.DAQmxWriteAnalogF64(
            self._ao_task, ctypes.c_int32(prefill),
            ctypes.c_uint32(0),
            ctypes.c_double(10.0),
            _VAL_GROUP_BY_CHAN,
            pre_buf, ctypes.byref(written), None,
        )
        print(f"[DAQ] WriteAnalogF64 prefill: err={err}, written={written.value}")
        if err != 0:
            self.error_msg = f"AO: pre-fill write failed ({err})"
            return False

        err = _dll.DAQmxStartTask(self._ao_task)
        print(f"[DAQ] StartTask AO: err={err}")
        if err != 0:
            self.error_msg = "AO: DAQmxStartTask failed"
            return False

        # ── AI ────────────────────────────────────────────────────────
        self._ai_task = ctypes.c_void_p(0)
        err = _dll.DAQmxCreateTask(b"", ctypes.byref(self._ai_task))
        print(f"[DAQ] CreateTask AI: err={err}, handle={self._ai_task.value}")
        if err != 0:
            self.error_msg = f"AI: DAQmxCreateTask failed ({err})"
            return False

        err = _dll.DAQmxCreateAIVoltageChan(
            self._ai_task, self.ai_channel, b"",
            _VAL_DIFF,
            ctypes.c_double(self.v_min),
            ctypes.c_double(self.v_max),
            _VAL_VOLTS, None,
        )
        print(f"[DAQ] CreateAIVoltageChan: err={err}")
        if err != 0:
            self.error_msg = f"AI: CreateAIVoltageChan failed ({err})"
            return False

        err = _dll.DAQmxCfgSampClkTiming(
            self._ai_task, b"",
            ctypes.c_double(float(FS)),
            _VAL_RISING, _VAL_CONT_SAMPS,
            ctypes.c_uint64(CHUNK * 20),
        )
        print(f"[DAQ] CfgSampClkTiming AI: err={err}")
        if err != 0:
            self.error_msg = f"AI: CfgSampClkTiming failed ({err})"
            return False

        err = _dll.DAQmxStartTask(self._ai_task)
        print(f"[DAQ] StartTask AI: err={err}")
        if err != 0:
            self.error_msg = "AI: DAQmxStartTask failed"
            return False

        self._running  = True
        self.connected = True
        threading.Thread(target=self._ao_loop, daemon=True).start()
        threading.Thread(target=self._ai_loop, daemon=True).start()
        return True

    # ── IO loops ───────────────────────────────────────────────────────

    def _generate_chunk(self):
        white          = np.random.normal(0, 1.0, CHUNK)
        base, self._ar_zi = lfilter([1], [1, -0.9], white, zi=self._ar_zi)
        return (self.amplitude * base
                if self._active else np.zeros(CHUNK)).astype(np.float64)

    def _ao_loop(self):
        written = ctypes.c_int32(0)
        while self._running:
            data    = self._generate_chunk()
            buf_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            err = _dll.DAQmxWriteAnalogF64(
                self._ao_task, ctypes.c_int32(CHUNK),
                ctypes.c_uint32(0),
                ctypes.c_double(2.0),
                _VAL_GROUP_BY_CHAN,
                buf_ptr, ctypes.byref(written), None,
            )
            if err < 0:
                print(f"[DAQ] AO write error: {err}")
                self.connected = False
                self._running  = False
                break

    def _ai_loop(self):
        ai_data = np.zeros(CHUNK, dtype=np.float64)
        buf_ptr = ai_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n_read  = ctypes.c_int32(0)
        while self._running:
            err = _dll.DAQmxReadAnalogF64(
                self._ai_task, ctypes.c_int32(CHUNK),
                ctypes.c_double(2.0),
                _VAL_GROUP_BY_CHAN,
                buf_ptr, ctypes.c_uint32(CHUNK),
                ctypes.byref(n_read), None,
            )
            if err < 0:
                print(f"[DAQ] AI read error: {err}")
                self.connected = False
                self._running  = False
                break
            with self._lock:
                self._buffer.extend(ai_data[:n_read.value].tolist())
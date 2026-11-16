"""GPU load and memory per process (NVIDIA, through NVML) for the header stats. Empty when there's no NVIDIA GPU."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger("theeye.gpu")

_lock = threading.Lock()
_cache: dict = {"t": 0.0, "v": None}
_nvml = {"ok": None, "h": None}


def _init() -> bool:
    if _nvml["ok"] is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml["h"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml["ok"] = True
        except Exception:  # noqa: BLE001
            log.info("no NVIDIA GPU stats (NVML unavailable)")
            _nvml["ok"] = False
    return bool(_nvml["ok"])


def _name(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


def stats(max_age: float = 2.0) -> Optional[dict]:
    """{name, util, mem_used, mem_total, temp, theeye, procs:[{pid, name, mem}]}, cached for max_age seconds."""
    with _lock:
        if time.time() - _cache["t"] < max_age:
            return _cache["v"]
        v = None
        if _init():
            try:
                import pynvml
                h = _nvml["h"]
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                procs: dict[int, int] = {}
                for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                    for p in fn(h):      # a process using CUDA and drawing shows up in both lists
                        procs[p.pid] = max(procs.get(p.pid, 0), p.usedGpuMemory or 0)
                me = os.getpid()
                name = pynvml.nvmlDeviceGetName(h)
                v = {"name": name.decode() if isinstance(name, bytes) else name,
                     "util": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                     "mem_used": mem.used, "mem_total": mem.total,
                     "temp": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                     "theeye": procs.get(me, 0),
                     "procs": sorted(({"pid": pid, "name": "TheEye" if pid == me else _name(pid), "mem": m}
                                      for pid, m in procs.items()), key=lambda p: -p["mem"])}
            except Exception:  # noqa: BLE001
                log.debug("GPU stats failed", exc_info=True)
        _cache.update(t=time.time(), v=v)
        return v

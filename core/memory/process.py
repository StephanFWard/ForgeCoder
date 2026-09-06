"""Process/system memory measurement (psutil with a stdlib fallback)."""
from __future__ import annotations

import os
import sys

try:
    import psutil
except ImportError:  # pragma: no cover - normal on minimal environments
    psutil = None

_MB = 1024 * 1024


def _win_global_memory() -> tuple[float, float] | None:
    """Return (total_gb, available_gb) using ctypes on Windows (no psutil)."""
    if os.name != "nt" or sys.platform.startswith("java"):
        return None
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    if not ok:
        return None
    return (stat.ullTotalPhys / _MB / 1024, stat.ullAvailPhys / _MB / 1024)


def total_ram_gb() -> float:
    if psutil is not None:
        return psutil.virtual_memory().total / _MB / 1024
    win = _win_global_memory()
    return win[0] if win else 8.0


def available_ram_gb() -> float:
    if psutil is not None:
        return psutil.virtual_memory().available / _MB / 1024
    win = _win_global_memory()
    return win[1] if win else 6.0


def cpu_cores() -> int:
    return os.cpu_count() or 4


def system_memory_pressure() -> float:
    """Used fraction of physical RAM (0..1)."""
    if psutil is not None:
        vm = psutil.virtual_memory()
        return vm.used / max(vm.total, 1)
    win = _win_global_memory()
    if win:
        total, avail = win
        return max(0.0, min(1.0, 1.0 - avail / max(total, 1.0)))
    return 0.6


def memory_usage_mb(pid: int | None = None) -> float:
    """RSS in MB for the current process or a given PID."""
    pid = pid or os.getpid()
    if psutil is not None:
        try:
            return psutil.Process(pid).memory_info().rss / _MB
        except psutil.Error:
            return 0.0
    return 0.0


def gpu_vram_gb() -> float:
    """Detect NVIDIA GPU VRAM via nvidia-smi (best-effort, 0 when absent)."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0.0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=False,
        ).stdout.strip()
        return float(out.splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0.0

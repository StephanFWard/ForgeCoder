"""Memory-derived budgets.

Hard constraints (from the master plan):

  Rule 5  — 4096-token maximum chat context
  Rule 6  — 1024-token maximum completion context
"""
from __future__ import annotations

from core.memory.process import cpu_cores, gpu_vram_gb, total_ram_gb


def context_budget_for_ram(ram_gb: float, *, max_context: int = 4096) -> int:
    """CPU-only chat context: 4096 for >= 6 GB, scaled down below that."""
    if ram_gb >= 6:
        return max_context
    if ram_gb >= 4:
        return 2048
    return 1024


def completion_budget_for_ram(ram_gb: float, *, max_completion: int = 1024) -> int:
    if ram_gb >= 6:
        return max_completion
    return 512 if ram_gb >= 4 else 256


def hardware_profile() -> dict:
    """Build the initial `%LOCALAPPDATA%/ForgeCoder/config.json` shape.

    Provisioning uses *total* RAM (a stable hardware property); momentary
    *available* RAM is the memory monitor's job at runtime.
    """
    ram = total_ram_gb()
    cores = cpu_cores()
    vram = gpu_vram_gb()
    gpu_layers = 19 if vram >= 4 else 0  # ~all layers of a 1.5B on 4 GB VRAM
    return {
        "ram_gb": round(ram, 1),
        "threads": cpu_threads_for(ram, cores),
        "context": context_budget_for_ram(ram),
        "batch": 256 if ram >= 6 else 128,
        "gpu_layers": gpu_layers,
        "completion_context": completion_budget_for_ram(ram),
    }


def cpu_threads_for(ram_gb: float, cores: int, *, reserve_for_gui: int = 2) -> int:
    """Leave a couple of cores free so the editor stays responsive."""
    if ram_gb < 4:
        return max(1, cores // 2)
    return max(1, cores - reserve_for_gui)

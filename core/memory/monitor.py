"""Periodic memory monitoring and escalation.

Escalation ladder (called from worst to best):

    clear retrieval cache
    disable background indexing
    reduce batch
    reduce context
"""
from __future__ import annotations

from dataclasses import dataclass

from core.memory.process import available_ram_gb, memory_usage_mb, system_memory_pressure

HIGH_PRESSURE = 0.85
CRITICAL_PRESSURE = 0.95


@dataclass(frozen=True)
class MemorySnapshot:
    pressure: float          # 0..1 used RAM
    available_gb: float
    model_mb: float
    server_mb: float
    actions: list[str]
    ok: bool


def _escalations(pressure: float) -> list[str]:
    if pressure >= CRITICAL_PRESSURE:
        return ["clear_retrieval_cache", "disable_background_indexing",
                "reduce_batch", "reduce_context"]
    if pressure >= HIGH_PRESSURE:
        return ["clear_retrieval_cache", "disable_background_indexing"]
    return []


class MemoryMonitor:
    def __init__(self, *, high: float = HIGH_PRESSURE, critical: float = CRITICAL_PRESSURE,
                 model_pid: int | None = None):
        self.high = high
        self.critical = critical
        self.model_pid = model_pid

    def snapshot(self, *, server_pid: int | None = None) -> MemorySnapshot:
        pressure = system_memory_pressure()
        avail = available_ram_gb()
        actions = _escalations(pressure)
        return MemorySnapshot(
            pressure=pressure,
            available_gb=avail,
            model_mb=memory_usage_mb(self.model_pid),
            server_mb=memory_usage_mb(server_pid),
            actions=actions,
            ok=not actions,
        )

    def should_throttle(self) -> bool:
        return system_memory_pressure() >= self.high

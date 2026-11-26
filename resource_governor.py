"""Limites cooperativos de CPU/RAM para treinos longos em CPU."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import torch


@dataclass
class ResourcePolicy:
    cpu_limit_percent: float = 75.0
    ram_limit_percent: float = 60.0
    poll_seconds: float = 2.0
    sleep_seconds: float = 1.0


class ResourceGovernor:
    """Governador cooperativo: pausa entre batches e não interrompe um batch."""

    def __init__(self, policy: ResourcePolicy | None = None):
        self.policy = policy or ResourcePolicy()
        self._last_process_cpu = time.process_time()
        self._last_wall = time.perf_counter()

    @staticmethod
    def ram_percent() -> float:
        try:
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint), ("memory_load", ctypes.c_uint),
                            ("total_phys", ctypes.c_size_t), ("avail_phys", ctypes.c_size_t),
                            ("total_page", ctypes.c_size_t), ("avail_page", ctypes.c_size_t),
                            ("total_virtual", ctypes.c_size_t), ("avail_virtual", ctypes.c_size_t),
                            ("avail_extended", ctypes.c_size_t)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.memory_load)
        except Exception:
            return 0.0

    def process_cpu_percent(self) -> float:
        now = time.perf_counter()
        process = time.process_time()
        wall_delta = max(now - self._last_wall, 1e-6)
        cpu_delta = process - self._last_process_cpu
        self._last_wall, self._last_process_cpu = now, process
        logical = max(os.cpu_count() or 1, 1)
        return max(0.0, min(100.0, 100.0 * cpu_delta / wall_delta / logical))

    def throttle(self) -> dict[str, float | bool]:
        cpu = self.process_cpu_percent()
        ram = self.ram_percent()
        paused = False
        while (cpu > self.policy.cpu_limit_percent or ram > self.policy.ram_limit_percent):
            paused = True
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(self.policy.sleep_seconds)
            cpu = self.process_cpu_percent()
            ram = self.ram_percent()
        return {"cpu_percent": cpu, "ram_percent": ram, "paused": paused}

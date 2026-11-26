"""Governador de recursos cooperativo com suporte híbrido Windows/Linux (WSL).

Monitora CPU e memória RAM durante o treinamento e auditoria, implementando
pausas cooperativas para evitar saturação do sistema operacional.
Em Linux/WSL2, obtém dados de memória via /proc/meminfo quando ctypes.windll
não está disponível.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

try:
    import torch
    # Invariante arquitetural: restringe execução em CPU a 2-4 threads
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
except Exception:
    pass


@dataclass
class ResourcePolicy:
    cpu_limit_percent: float = 75.0
    ram_limit_percent: float = 60.0
    poll_seconds: float = 2.0
    sleep_seconds: float = 1.0


class SandboxResourceGovernor:
    """Governador cooperativo para auditoria e treino no sandbox."""

    def __init__(self, policy: ResourcePolicy | None = None, num_threads: int = 2):
        self.policy = policy or ResourcePolicy()
        self.num_threads = num_threads
        self._last_process_cpu = time.process_time()
        self._last_wall = time.perf_counter()

        # Garante a política de contenção de threads PyTorch
        try:
            import torch
            torch.set_num_threads(max(1, min(self.num_threads, 4)))
        except Exception:
            pass

    @staticmethod
    def ram_percent() -> float:
        """Calcula percentual de uso de RAM com suporte dual Windows e Linux."""
        # 1. Tentativa Windows via GlobalMemoryStatusEx
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint),
                    ("memory_load", ctypes.c_uint),
                    ("total_phys", ctypes.c_size_t),
                    ("avail_phys", ctypes.c_size_t),
                    ("total_page", ctypes.c_size_t),
                    ("avail_page", ctypes.c_size_t),
                    ("total_virtual", ctypes.c_size_t),
                    ("avail_virtual", ctypes.c_size_t),
                    ("avail_extended", ctypes.c_size_t),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.memory_load)
        except Exception:
            pass

        # 2. Fallback POSIX / Linux via /proc/meminfo
        try:
            meminfo: dict[str, float] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = float(parts[1].split()[0])
            total = meminfo.get("MemTotal", 1.0)
            avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 1.0))
            return max(0.0, min(100.0, float((1.0 - (avail / total)) * 100.0)))
        except Exception:
            return 0.0

    def process_cpu_percent(self) -> float:
        """Calcula percentual de CPU do processo normalizado pelo número de núcleos lógicos."""
        now = time.perf_counter()
        process = time.process_time()
        wall_delta = max(now - self._last_wall, 1e-6)
        cpu_delta = process - self._last_process_cpu
        self._last_wall, self._last_process_cpu = now, process
        logical = max(os.cpu_count() or 1, 1)
        return max(0.0, min(100.0, 100.0 * cpu_delta / wall_delta / logical))

    def throttle(self) -> dict[str, float | bool]:
        """Pausa cooperativamente se o uso de CPU ou RAM exceder os limites da política."""
        cpu = self.process_cpu_percent()
        ram = self.ram_percent()
        paused = False
        while cpu > self.policy.cpu_limit_percent or ram > self.policy.ram_limit_percent:
            paused = True
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(self.policy.sleep_seconds)
            cpu = self.process_cpu_percent()
            ram = self.ram_percent()
        return {"cpu_percent": cpu, "ram_percent": ram, "paused": paused}


# Alias para compatibilidade com código que importa ResourceGovernor diretamente
ResourceGovernor = SandboxResourceGovernor

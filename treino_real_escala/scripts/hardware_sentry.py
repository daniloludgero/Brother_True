"""Sentinela de Hardware, Memória RAM e Proteção do Disco SSD C:.

Monitora ativamente o espaço livre em disco, memória RAM e temperatura de CPU.
Impede qualquer saturação do hardware local:
- Garante que o SSD C: permaneça sempre com mais de 4.5 GB livres.
- Aciona expurgo automático de caches temporários (.hf_cache, /tmp/) se o espaço cair.
- Pausa cooperativamente o treinamento se a RAM exceder 65% ou a CPU exceder 75%.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import torch


@dataclass
class SentryPolicy:
    cpu_limit_percent: float = 75.0
    ram_limit_percent: float = 65.0
    min_disk_free_gb: float = 4.5
    emergency_disk_floor_gb: float = 4.0
    sleep_seconds: float = 1.0


class HardwareSentry:
    """Sentinela autônomo para proteção integral do hardware durante treinos longos."""

    def __init__(self, policy: SentryPolicy | None = None, num_threads: int = 2):
        self.policy = policy or SentryPolicy()
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

        self._last_process_cpu = time.process_time()
        self._last_wall = time.perf_counter()

    @staticmethod
    def disk_free_gb(path: str = "/mnt/c") -> float:
        """Retorna os Gigabytes livres na partição especificada."""
        try:
            stat = shutil.disk_usage(path)
            return stat.free / (1024 ** 3)
        except Exception:
            try:
                stat = shutil.disk_usage(".")
                return stat.free / (1024 ** 3)
            except Exception:
                return 10.0

    @staticmethod
    def ram_percent() -> float:
        """Aferição precisa de RAM sob Linux / WSL2 via /proc/meminfo ou Windows API."""
        if os.path.exists("/proc/meminfo"):
            try:
                mem_total = 0
                mem_avail = 0
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_total = int(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            mem_avail = int(line.split()[1])
                if mem_total > 0:
                    used = mem_total - mem_avail
                    return (used / mem_total) * 100.0
            except Exception:
                pass

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
            return 40.0

    def process_cpu_percent(self) -> float:
        """Aferição da taxa de CPU normalizada pelo total de núcleos lógicos."""
        now = time.perf_counter()
        process = time.process_time()
        wall_delta = max(now - self._last_wall, 1e-6)
        cpu_delta = process - self._last_process_cpu
        self._last_wall, self._last_process_cpu = now, process
        logical = max(os.cpu_count() or 1, 1)
        return max(0.0, min(100.0, 100.0 * cpu_delta / wall_delta / logical))

    def purge_caches(self):
        """Expurga agressivamente caches temporários em caso de pressão de disco."""
        targets = [
            Path(".hf_cache"),
            Path.home() / ".cache" / "huggingface",
            Path("/tmp") / "huggingface",
        ]
        purged_bytes = 0
        for t in targets:
            if t.exists():
                try:
                    if t.is_dir():
                        size = sum(f.stat().st_size for f in t.glob("**/*") if f.is_file())
                        shutil.rmtree(t, ignore_errors=True)
                        purged_bytes += size
                    else:
                        purged_bytes += t.stat().st_size
                        t.unlink(missing_ok=True)
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return purged_bytes

    def throttle(self) -> Dict[str, Union[float, bool]]:
        """Aplica pausas cooperativas e salvaguardas de disco e memória."""
        free_gb = self.disk_free_gb()

        # Proteção emergencial de disco
        if free_gb < self.policy.min_disk_free_gb:
            print(f"[Sentinela Hardware] Espaço livre ({free_gb:.2f} GB) abaixo de {self.policy.min_disk_free_gb} GB! Acionando expurgo...")
            self.purge_caches()
            free_gb = self.disk_free_gb()

        if free_gb < self.policy.emergency_disk_floor_gb:
            raise RuntimeError(
                f"[Sentinela Hardware - ERRO CRÍTICO] Espaço em disco perigosamente baixo ({free_gb:.2f} GB < {self.policy.emergency_disk_floor_gb} GB). "
                f"Execução bloqueada para proteger o SSD host."
            )

        cpu = self.process_cpu_percent()
        ram = self.ram_percent()
        paused = False

        while cpu > self.policy.cpu_limit_percent or ram > self.policy.ram_limit_percent:
            paused = True
            gc.collect()
            time.sleep(self.policy.sleep_seconds)
            cpu = self.process_cpu_percent()
            ram = self.ram_percent()

        return {
            "cpu_percent": round(cpu, 1),
            "ram_percent": round(ram, 1),
            "disk_free_gb": round(free_gb, 2),
            "paused": paused,
        }


if __name__ == "__main__":
    sentry = HardwareSentry()
    status = sentry.throttle()
    print(f"[Sentinela Status Inicial] RAM: {status['ram_percent']}% | CPU: {status['cpu_percent']}% | SSD Livre: {status['disk_free_gb']} GB")

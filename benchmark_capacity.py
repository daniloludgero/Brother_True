"""Dimensiona a capacidade de treino da Voz com uma etapa por vez.

O benchmark usa apenas dados sintéticos para medir arquitetura e memória.
Nenhum texto dos CSVs é executado ou interpretado como código.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from ctypes import Structure, byref, c_ulong, sizeof, windll
from pathlib import Path

import torch

from .voz import Voz


class _MemoryCounters(Structure):
    _fields_ = [
        ("cb", c_ulong),
        ("PageFaultCount", c_ulong),
        ("PeakWorkingSetSize", c_ulong),
        ("WorkingSetSize", c_ulong),
        ("QuotaPeakPagedPoolUsage", c_ulong),
        ("QuotaPagedPoolUsage", c_ulong),
        ("QuotaPeakNonPagedPoolUsage", c_ulong),
        ("QuotaNonPagedPoolUsage", c_ulong),
        ("PagefileUsage", c_ulong),
        ("PeakPagefileUsage", c_ulong),
    ]


def _rss_bytes() -> int:
    counters = _MemoryCounters()
    counters.cb = sizeof(counters)
    windll.psapi.GetProcessMemoryInfo(-1, byref(counters), sizeof(counters))
    return int(counters.WorkingSetSize)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _candidate_configs() -> list[dict[str, int]]:
    return [
        {"d_model": 256, "nhead": 4, "num_layers": 3, "dim_feedforward": 512},
        {"d_model": 384, "nhead": 6, "num_layers": 4, "dim_feedforward": 1536},
        {"d_model": 512, "nhead": 8, "num_layers": 4, "dim_feedforward": 2048},
        {"d_model": 512, "nhead": 8, "num_layers": 6, "dim_feedforward": 2048},
        {"d_model": 768, "nhead": 12, "num_layers": 6, "dim_feedforward": 3072},
    ]


def benchmark(config: dict[str, int], vocab_size: int, batch_size: int, seq_len: int) -> dict:
    before = _rss_bytes()
    model = Voz(vocab_size=vocab_size, max_len=seq_len, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    humor = torch.randn(batch_size, 11)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids, humor)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - started
    peak = _rss_bytes()

    result = {
        **config,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parameters": _count(model),
        "loss": float(loss.detach()),
        "step_seconds": round(elapsed, 3),
        "rss_before_mb": round(before / 2**20, 1),
        "rss_after_mb": round(peak / 2**20, 1),
        "rss_delta_mb": round((peak - before) / 2**20, 1),
        "status": "ok",
    }
    del optimizer, model, input_ids, humor, target, logits, loss
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/capacity.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    results = []
    for config in _candidate_configs():
        try:
            result = benchmark(config, args.vocab_size, args.batch_size, args.seq_len)
        except (RuntimeError, MemoryError) as exc:
            result = {**config, "status": "failed", "error": str(exc)[:500]}
            gc.collect()
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

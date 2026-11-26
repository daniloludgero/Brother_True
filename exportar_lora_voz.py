"""Exporta somente os pesos LoRA de um checkpoint completo."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_adapter(source: Path, target: Path) -> None:
    state = torch.load(source, map_location="cpu", weights_only=False)
    model_state = state["model"]
    adapter_state = {
        key: value.cpu()
        for key, value in model_state.items()
        if ".lora_a" in key or ".lora_b" in key
    }
    if not adapter_state:
        raise RuntimeError(f"Nenhum peso LoRA encontrado em {source}")
    metrics = state.get("metrics", {})
    payload = {
        "format": "brothertrue-lora-v1",
        "base_checkpoint": metrics.get("base"),
        "rank": metrics.get("rank", 8),
        "alpha": 16.0,
        "total_parameters": metrics.get("total_parameters"),
        "trainable_parameters": sum(t.numel() for t in adapter_state.values()),
        "step": state.get("step", 0),
        "metrics": metrics,
        "adapter_state_dict": adapter_state,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    print(f"[OK] Adaptador exportado: {target} ({target.stat().st_size} bytes)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    a = p.parse_args()
    export_adapter(a.source, a.target)


if __name__ == "__main__":
    main()

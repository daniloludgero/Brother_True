"""Checkpoints atômicos para retomada de treinos interrompidos."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: str | os.PathLike, *, model, optimizer=None, scheduler=None,
                    epoch: int = 0, step: int = 0, metrics: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    fd, temporary = tempfile.mkstemp(prefix=target.stem + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | os.PathLike, *, model, optimizer=None, scheduler=None,
                    map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if payload.get("python_rng") is not None:
        random.setstate(payload["python_rng"])
    if payload.get("numpy_rng") is not None:
        np.random.set_state(payload["numpy_rng"])
    if payload.get("torch_rng") is not None:
        torch.set_rng_state(payload["torch_rng"])
    return {"epoch": payload.get("epoch", 0), "step": payload.get("step", 0),
            "metrics": payload.get("metrics", {})}

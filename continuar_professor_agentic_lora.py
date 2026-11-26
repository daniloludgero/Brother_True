"""Continuação do LoRA do professor agentic a partir do melhor piloto."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .resource_governor import ResourceGovernor, ResourcePolicy
from .treinar_professor_agentic_lora import (
    add_lora, adapter_state, encode, evaluate, read_rows,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-every", type=int, default=20)
    a = p.parse_args()
    random.seed(20260825)
    torch.manual_seed(20260825)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = read_rows(a.parquet, a.rows)
    random.shuffle(rows)
    split = max(1, len(rows) // 10)
    val, train = rows[:split], rows[split:]
    tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(a.teacher), local_files_only=True, dtype=torch.float32,
        low_cpu_mem_usage=True)
    model.eval()
    trainable = add_lora(model, a.rank, a.rank * 2.0)
    previous = torch.load(a.resume, map_location="cpu")
    missing, unexpected = model.load_state_dict(previous["adapter"], strict=False)
    total = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90,
                                                sleep_seconds=1.0))
    best = float("inf")
    print(json.dumps({"event": "continuation_initialized", "rows": len(rows),
                      "train_rows": len(train), "val_rows": len(val),
                      "total_parameters": total, "trainable_parameters": trainable,
                      "resume": str(a.resume), "missing_adapter_keys": len(missing),
                      "unexpected_keys": len(unexpected), "lr": a.lr}, ensure_ascii=False), flush=True)
    for step in range(1, a.steps + 1):
        row = train[(step * 104729) % len(train)]
        x, y = encode(row, tok, a.max_len)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=x.unsqueeze(0)).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y, ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        resources = governor.throttle()
        if step % a.save_every == 0 or step == a.steps:
            val_loss = evaluate(model, val, tok, a.max_len)
            payload = {"step": step, "train_loss": float(loss.detach()),
                       "val_loss": val_loss, "rows": len(rows),
                       "total_parameters": total, "trainable_parameters": trainable,
                       "target": "final_answer+tool_calls", "base": str(a.teacher),
                       "resume": str(a.resume)}
            torch.save({"adapter": adapter_state(model), "metadata": payload},
                       a.out / "continuacao_latest.pt")
            if val_loss < best:
                best = val_loss
                torch.save({"adapter": adapter_state(model), "metadata": payload},
                           a.out / "continuacao_best.pt")
            print(json.dumps({"event": "continuation_checkpoint", **payload,
                              "best_val": best, **resources}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "continuation_complete", "best_val": best,
                      "out": str(a.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

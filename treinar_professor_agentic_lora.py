"""Treino LoRA curto do professor agentic Qwen a partir do Parquet local."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .resource_governor import ResourceGovernor, ResourcePolicy


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return self.base(x) + update


def add_lora(model, rank: int, alpha: float) -> int:
    for p in model.parameters():
        p.requires_grad = False
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name.endswith(".q_proj") or name.endswith(".v_proj") or
            name.endswith(".o_proj") or name == "lm_head"):
            parent_name, child = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            targets.append((parent, child, module))
    trainable = 0
    for parent, child, base in targets:
        wrapped = LoRALinear(base, rank, alpha)
        setattr(parent, child, wrapped)
        trainable += wrapped.lora_A.numel() + wrapped.lora_B.numel()
    return trainable


def compact_target(raw: str) -> str | None:
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    target = {"final_answer": obj.get("final_answer", ""),
              "tool_calls": obj.get("tool_calls", [])}
    if not target["final_answer"] and not target["tool_calls"]:
        return None
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=["prompt", "response", "kind"]):
        for row in batch.to_pylist():
            target = compact_target(row.get("response", ""))
            if target and row.get("prompt"):
                rows.append({"prompt": row["prompt"], "target": target,
                             "kind": row.get("kind", "")})
                if len(rows) >= limit:
                    return rows
    return rows


def encode(row, tok, max_len: int):
    marker = "\n\nASSISTANT_JSON:\n"
    prompt_ids = tok.encode(row["prompt"] + marker, add_special_tokens=False)
    target_ids = tok.encode(row["target"], add_special_tokens=False)
    eos = tok.eos_token_id
    target_ids = (target_ids + ([eos] if eos is not None else []))[:max_len - 2]
    prompt_ids = prompt_ids[-max(1, max_len - len(target_ids) - 1):]
    ids = prompt_ids + target_ids
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[:max(0, len(prompt_ids) - 1)] = -100
    return x, y


@torch.no_grad()
def evaluate(model, rows, tok, max_len):
    model.eval()
    values = []
    for row in rows[:16]:
        x, y = encode(row, tok, max_len)
        logits = model(input_ids=x.unsqueeze(0)).logits
        values.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            y, ignore_index=-100)))
    return sum(values) / max(1, len(values))


def adapter_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save-every", type=int, default=10)
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
    total = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90,
                                                sleep_seconds=1.0))
    best = float("inf")
    print(json.dumps({"event": "initialized", "rows": len(rows),
                      "train_rows": len(train), "val_rows": len(val),
                      "total_parameters": total, "trainable_parameters": trainable,
                      "teacher_base": str(a.teacher), "rank": a.rank,
                      "target": "final_answer+tool_calls", "max_len": a.max_len}, ensure_ascii=False), flush=True)
    for step in range(1, a.steps + 1):
        row = train[(step * 7919) % len(train)]
        x, y = encode(row, tok, a.max_len)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=x.unsqueeze(0)).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y, ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        metrics = governor.throttle()
        if step % a.save_every == 0 or step == a.steps:
            val_loss = evaluate(model, val, tok, a.max_len)
            payload = {"step": step, "train_loss": float(loss.detach()),
                       "val_loss": val_loss, "rows": len(rows),
                       "total_parameters": total, "trainable_parameters": trainable,
                       "target": "final_answer+tool_calls", "base": str(a.teacher)}
            torch.save({"adapter": adapter_state(model), "metadata": payload}, a.out / "latest.pt")
            if val_loss < best:
                best = val_loss
                torch.save({"adapter": adapter_state(model), "metadata": payload}, a.out / "best.pt")
            print(json.dumps({"event": "checkpoint", **payload, "best_val": best,
                              **metrics}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "complete", "best_val": best,
                      "out": str(a.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

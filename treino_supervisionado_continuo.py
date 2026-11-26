"""Continuação do treino mantendo somente um checkpoint principal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .voz import Voz


def stream_examples(path, tokenizer, max_len):
    with path.open(encoding="utf-8") as source:
        for line in source:
            ids = tokenizer.encode(json.loads(line).get("text", "")).ids[:max_len]
            if len(ids) >= 2:
                yield ids[:-1], ids[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = Voz(vocab_size=tokenizer.get_vocab_size(), d_model=384, nhead=6,
                num_layers=4, dim_feedforward=1536, max_len=args.max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    start = 0
    if latest.exists():
        state = load_checkpoint(latest, model=model, optimizer=optimizer, map_location="cpu")
        start = int(state["step"])
        print(f"[OK] Retomando do passo {start}", flush=True)
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=90.0))
    iterator = stream_examples(args.data, tokenizer, args.max_len)
    for step in range(start, args.steps):
        try:
            a, b = next(iterator)
        except StopIteration:
            iterator = stream_examples(args.data, tokenizer, args.max_len)
            a, b = next(iterator)
        x = torch.tensor([a], dtype=torch.long)
        y = torch.tensor([b], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, torch.zeros(1, 11))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0 or step == args.steps - 1:
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step + 1,
                            metrics={"loss": float(loss.detach())})
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()),
                              **governor.throttle()}, ensure_ascii=False), flush=True)
    print(f"[OK] Treino contínuo concluído no passo {args.steps}", flush=True)


if __name__ == "__main__":
    main()

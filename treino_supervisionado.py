"""Treino piloto da Voz por próxima palavra, sem depender de logits externos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .voz import Voz


def exemplos(path: Path, tokenizer: Tokenizer, max_len: int):
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            ids = tokenizer.encode(item.get("text", "")).ids[:max_len]
            if len(ids) >= 2:
                yield ids[:-1], ids[1:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dados_preparados/dataset_normalizado.jsonl"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer_voz_16k.json"))
    p.add_argument("--out", type=Path, default=Path("checkpoints_voz_supervisionado"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args()

    torch.set_num_threads(min(2, torch.get_num_threads()))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = Voz(vocab_size=tokenizer.get_vocab_size(), d_model=384, nhead=6,
                num_layers=4, dim_feedforward=1536, max_len=args.max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=90.0))
    args.out.mkdir(parents=True, exist_ok=True)

    latest = args.out / "latest.pt"
    start_step = 0
    if latest.exists():
        state = load_checkpoint(latest, model=model, optimizer=optimizer, map_location="cpu")
        start_step = int(state["step"])
        print(f"[OK] Retomando do passo {start_step}", flush=True)

    iterator = exemplos(args.data, tokenizer, args.max_len)
    for step in range(start_step, args.steps):
        try:
            input_ids, target_ids = next(iterator)
        except StopIteration:
            iterator = exemplos(args.data, tokenizer, args.max_len)
            input_ids, target_ids = next(iterator)
        x = torch.tensor([input_ids], dtype=torch.long)
        y = torch.tensor([target_ids], dtype=torch.long)
        humor = torch.zeros(1, 11)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, humor)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5 == 0 or step == args.steps - 1:
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step + 1,
                            metrics={"loss": float(loss.detach())})
            save_checkpoint(args.out / f"voz_step_{step + 1:06d}.pt", model=model,
                            optimizer=optimizer, step=step + 1,
                            metrics={"loss": float(loss.detach())})
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()),
                              **governor.throttle()}, ensure_ascii=False), flush=True)

    print(f"[OK] Treino piloto concluído: {args.steps} passos", flush=True)


if __name__ == "__main__":
    main()

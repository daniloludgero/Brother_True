"""Continuação com learning rate reduzido, melhor checkpoint e early stopping."""

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
from .treino_curriculo import build_offsets, evaluate, read_batch
from .voz import Voz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1250)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    pad_id = tokenizer.token_to_id("<pad>") or 0
    offsets = build_offsets(args.data)
    indices = list(range(len(offsets)))
    random.Random(args.seed).shuffle(indices)
    val_count = max(1, int(len(indices) * 0.10))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    model = Voz(vocab_size=tokenizer.get_vocab_size(), d_model=384, nhead=6,
                num_layers=4, dim_feedforward=1536, max_len=args.max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    best = args.out / "best.pt"
    start = 0
    if latest.exists():
        state = load_checkpoint(latest, model=model, optimizer=optimizer, map_location="cpu")
        start = int(state["step"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        print(f"[OK] Retomando do passo {start} com lr={args.lr}", flush=True)

    best_val = float("inf")
    if best.exists():
        best_state = load_checkpoint(best, model=model, map_location="cpu")
        best_val = float(best_state["metrics"].get("val_loss", float("inf")))
        load_checkpoint(latest, model=model, optimizer=optimizer, map_location="cpu")
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=90.0))
    order = train_indices[:]
    random.Random(args.seed + start).shuffle(order)
    cursor = 0
    no_improve = 0
    for step in range(start, args.steps):
        if cursor + args.batch_size > len(order):
            random.shuffle(order)
            cursor = 0
        batch = read_batch(args.data, offsets, order[cursor:cursor + args.batch_size],
                           tokenizer, args.max_len, pad_id)
        cursor += args.batch_size
        if batch is None:
            continue
        x, y = batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, torch.zeros(x.size(0), 11), tgt_key_padding_mask=(x == pad_id))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == args.steps - 1:
            val_loss = evaluate(model, args.data, offsets, val_indices, tokenizer,
                                args.max_len, pad_id, args.batch_size)
            metric = {"train_loss": float(loss.detach()), "val_loss": val_loss,
                      "train_examples": len(train_indices), "val_examples": len(val_indices),
                      "learning_rate": args.lr}
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step + 1, metrics=metric)
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                save_checkpoint(best, model=model, optimizer=optimizer, step=step + 1, metrics=metric)
            else:
                no_improve += 1
            print(json.dumps({"step": step + 1, **metric, "best_val": best_val,
                              "no_improve": no_improve, **governor.throttle()}, ensure_ascii=False), flush=True)
            if no_improve >= args.patience:
                print(f"[STOP] Early stopping no passo {step + 1}", flush=True)
                break
    print(f"[OK] Treino adaptativo concluído. Melhor validação: {best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()

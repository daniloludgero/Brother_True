"""Treino da Voz com embaralhamento, split de validação e batches reais."""

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


def build_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open("rb") as source:
        while True:
            position = source.tell()
            line = source.readline()
            if not line:
                break
            if line.strip():
                offsets.append(position)
    return offsets


def read_batch(path: Path, offsets: list[int], indices: list[int], tokenizer: Tokenizer,
               max_len: int, pad_id: int):
    inputs, targets = [], []
    with path.open("rb") as source:
        for index in indices:
            source.seek(offsets[index])
            item = json.loads(source.readline().decode("utf-8"))
            ids = tokenizer.encode(item.get("text", "")).ids[:max_len]
            if len(ids) >= 2:
                inputs.append(torch.tensor(ids[:-1], dtype=torch.long))
                targets.append(torch.tensor(ids[1:], dtype=torch.long))
    if not inputs:
        return None
    x = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    y = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_id)
    return x, y


@torch.no_grad()
def evaluate(model, path, offsets, indices, tokenizer, max_len, pad_id, batch_size, limit=20):
    model.eval()
    losses = []
    for start in range(0, min(len(indices), limit * batch_size), batch_size):
        batch = read_batch(path, offsets, indices[start:start + batch_size], tokenizer, max_len, pad_id)
        if batch is None:
            continue
        x, y = batch
        logits = model(x, torch.zeros(x.size(0), 11),
                       tgt_key_padding_mask=(x == pad_id))
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                            ignore_index=pad_id)))
    return sum(losses) / max(len(losses), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.10)
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
    val_count = max(1, int(len(indices) * args.val_fraction))
    val_indices, train_indices = indices[:val_count], indices[val_count:]

    model = Voz(vocab_size=tokenizer.get_vocab_size(), d_model=384, nhead=6,
                num_layers=4, dim_feedforward=1536, max_len=args.max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    start_step = 0
    if latest.exists():
        state = load_checkpoint(latest, model=model, optimizer=optimizer, map_location="cpu")
        start_step = int(state["step"])
        print(f"[OK] Retomando do passo {start_step}", flush=True)

    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=90.0))
    train_rng = random.Random(args.seed + start_step)
    order = train_indices[:]
    cursor = 0
    for step in range(start_step, args.steps):
        if cursor + args.batch_size > len(order):
            train_rng.shuffle(order)
            cursor = 0
        selected = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        batch = read_batch(args.data, offsets, selected, tokenizer, args.max_len, pad_id)
        if batch is None:
            continue
        x, y = batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, torch.zeros(x.size(0), 11),
                       tgt_key_padding_mask=(x == pad_id))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == args.steps - 1:
            val_loss = evaluate(model, args.data, offsets, val_indices, tokenizer,
                                args.max_len, pad_id, args.batch_size)
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step + 1,
                            metrics={"train_loss": float(loss.detach()), "val_loss": val_loss,
                                     "train_examples": len(train_indices), "val_examples": len(val_indices)})
            print(json.dumps({"step": step + 1, "train_loss": float(loss.detach()),
                              "val_loss": val_loss, **governor.throttle()}, ensure_ascii=False), flush=True)
    print(f"[OK] Currículo concluído no passo {args.steps}", flush=True)


if __name__ == "__main__":
    main()

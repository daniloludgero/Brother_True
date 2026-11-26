"""Pré-treino limpo da Voz em linguagem natural, sem dados factuais/código."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
    from .resource_governor import ResourceGovernor, ResourcePolicy
    from .voz import Voz
except (ImportError, ValueError):
    from checkpoint_manager_safe import load_checkpoint, save_checkpoint
    from resource_governor import ResourceGovernor, ResourcePolicy
    from voz import Voz


def offsets(path: Path):
    positions = []
    with path.open("rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                positions.append(position)
    return positions


def sample(path, position, tok, max_len, bos, eos):
    with path.open("rb") as handle:
        handle.seek(position)
        item = json.loads(handle.readline().decode("utf-8"))
    ids = tok.encode(item["text"], add_special_tokens=False).ids
    if len(ids) < 2:
        return None
    seq = [bos] + ids + [eos]
    if len(seq) > max_len:
        start = random.randint(0, len(seq) - max_len)
        seq = seq[start:start + max_len]
        seq[0] = bos
    x = torch.tensor(seq[:-1], dtype=torch.long)
    y = torch.tensor(seq[1:], dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate(model, path, positions, tok, max_len, bos, eos):
    model.eval(); values = []
    for position in positions[:32]:
        item = sample(path, position, tok, max_len, bos, eos)
        if item is None:
            continue
        x, y = item
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        values.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y)))
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()
    random.seed(20260823); torch.manual_seed(20260823)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(args.tokenizer))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    positions = offsets(args.data)
    order = list(range(len(positions))); random.shuffle(order)
    val = order[:max(1, len(order) // 10)]
    train = order[len(val):]
    model = Voz(vocab_size=tok.get_vocab_size(), d_model=384, nhead=6,
                num_layers=4, dim_feedforward=1536, max_len=args.max_len,
                pad_token_id=pad, bos_token_id=bos, eos_token_id=eos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    latest, best = args.out / "latest.pt", args.out / "best.pt"
    step = 0; best_val = float("inf")
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,
                                                ram_limit_percent=90))
    while step < args.steps:
        position = positions[train[random.randrange(len(train))]]
        item = sample(args.data, position, tok, args.max_len, bos, eos)
        if item is None:
            continue
        x, y = item
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); step += 1
        if step % 25 == 0 or step == args.steps:
            val_loss = evaluate(model, args.data, [positions[i] for i in val], tok, args.max_len, bos, eos)
            metrics = {"train_loss": float(loss.detach()), "val_loss": val_loss,
                       "learning_rate": args.lr, "dataset": "voz_natural_only",
                       "vocab_size": tok.get_vocab_size()}
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step, metrics=metrics)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best, model=model, optimizer=optimizer, step=step, metrics=metrics)
            print(json.dumps({"step":step, **metrics, "best_val":best_val,
                              **governor.throttle()}, ensure_ascii=False), flush=True)
    print(f"[OK] Voz limpa concluída; melhor validação {best_val:.5f}", flush=True)


if __name__ == "__main__": main()

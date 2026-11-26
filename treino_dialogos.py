"""Fine-tuning contextual da Voz em pares prompt -> resposta."""

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


def offsets(path):
    result = []
    with path.open("rb") as f:
        while True:
            pos = f.tell(); line = f.readline()
            if not line: break
            if line.strip(): result.append(pos)
    return result


def make_batch(path, positions, tok, max_len, pad):
    xs, ys = [], []
    with path.open("rb") as f:
        for pos in positions:
            f.seek(pos); item = json.loads(f.readline().decode("utf-8"))
            prompt = tok.encode(item["prompt"]).ids
            response = tok.encode(item["response"]).ids
            prompt = prompt[-max_len // 2:]
            response = response[:max_len - len(prompt)]
            if len(prompt) < 1 or len(response) < 1: continue
            sequence = prompt + response
            xs.append(torch.tensor(sequence[:-1], dtype=torch.long))
            ys.append(torch.tensor([pad] * (len(prompt) - 1) + response, dtype=torch.long))
    if not xs: return None
    return (torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad),
            torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad))


@torch.no_grad()
def evaluate(model, path, pos, tok, max_len, pad, batch_size, limit=20):
    model.eval(); values = []
    for start in range(0, min(len(pos), limit * batch_size), batch_size):
        batch = make_batch(path, pos[start:start + batch_size], tok, max_len, pad)
        if batch is None: continue
        x, y = batch
        logits = model(x, torch.zeros(x.size(0), 11), tgt_key_padding_mask=(x == pad))
        values.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)))
    return sum(values) / max(1, len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True); p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=500); p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(args.tokenizer)); pad = tok.token_to_id("<pad>") or 0
    positions = offsets(args.data); order = list(range(len(positions))); random.Random(args.seed).shuffle(order)
    val_n = max(1, int(len(order) * .1)); val, train = order[:val_n], order[val_n:]
    model = Voz(vocab_size=tok.get_vocab_size(), d_model=384, nhead=6, num_layers=4, dim_feedforward=1536, max_len=args.max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr); args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"; best = args.out / "best.pt"; start = 0
    source = latest if latest.exists() else args.base
    state = load_checkpoint(source, model=model, optimizer=optimizer, map_location="cpu")
    start = int(state.get("step", 0)); print(f"[OK] Base contextual: {source.name}; passo {start}", flush=True)
    for group in optimizer.param_groups: group["lr"] = args.lr
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90)); best_val = float("inf")
    rng = random.Random(args.seed + start); cursor = 0
    while start < args.steps:
        if cursor >= len(train): rng.shuffle(train); cursor = 0
        batch = make_batch(args.data, [positions[train[cursor]]], tok, args.max_len, pad); cursor += 1
        if batch is None: continue
        x, y = batch; model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x, torch.zeros(x.size(0), 11), tgt_key_padding_mask=(x == pad))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); start += 1
        if start % 25 == 0 or start == args.steps:
            val_loss = evaluate(model, args.data, [positions[i] for i in val], tok, args.max_len, pad, 1)
            metrics = {"train_loss": float(loss.detach()), "val_loss": val_loss, "train_examples": len(train), "val_examples": len(val), "learning_rate": args.lr}
            save_checkpoint(latest, model=model, optimizer=optimizer, step=start, metrics=metrics)
            if val_loss < best_val:
                best_val = val_loss; save_checkpoint(best, model=model, optimizer=optimizer, step=start, metrics=metrics)
            print(json.dumps({"step": start, **metrics, "best_val": best_val, **governor.throttle()}, ensure_ascii=False), flush=True)
    print(f"[OK] Dialogo contextual concluído; melhor validação {best_val:.5f}", flush=True)


if __name__ == "__main__": main()

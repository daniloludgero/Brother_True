"""Fine-tuning com BOS/EOS controlados explicitamente."""

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
    out = []
    with path.open("rb") as f:
        while True:
            pos = f.tell(); line = f.readline()
            if not line: break
            if line.strip(): out.append(pos)
    return out


def batch(path, positions, tok, max_len, pad, bos, eos):
    xs, ys = [], []
    with path.open("rb") as f:
        for pos in positions:
            f.seek(pos); item = json.loads(f.readline().decode("utf-8"))
            prompt = tok.encode(item["prompt"], add_special_tokens=False).ids
            response = tok.encode(item["response"], add_special_tokens=False).ids
            prompt = prompt[-max_len // 2:]
            response = response[:max_len - len(prompt) - 1]
            if not prompt or not response: continue
            seq = [bos] + prompt + response + [eos]
            seq = seq[:max_len]
            x = seq[:-1]; target_start = len(prompt) + 1
            y = [pad] * target_start + seq[target_start + 1:]
            y = y[:len(x)]
            xs.append(torch.tensor(x)); ys.append(torch.tensor(y))
    if not xs: return None
    return torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad), torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad)


@torch.no_grad()
def evaluate(model, path, positions, tok, max_len, pad, bos, eos):
    model.eval(); vals = []
    for pos in positions[:20]:
        item = batch(path, [pos], tok, max_len, pad, bos, eos)
        if item is None: continue
        x, y = item; logits = model(x, torch.zeros(1, 11), tgt_key_padding_mask=(x == pad))
        vals.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)))
    return sum(vals) / max(1, len(vals))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--tokenizer", type=Path, required=True); p.add_argument("--base", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--steps", type=int, default=1800); p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args(); random.seed(20260823); torch.manual_seed(20260823); torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(args.tokenizer)); pad = tok.token_to_id("<pad>") or 0; bos = tok.token_to_id("<bos>") or 1; eos = tok.token_to_id("<eos>") or 2
    pos = offsets(args.data); order = list(range(len(pos))); random.shuffle(order); val = order[:max(1, len(order)//10)]; train = order[len(val):]
    model = Voz(vocab_size=tok.get_vocab_size(), d_model=384, nhead=6, num_layers=4, dim_feedforward=1536, max_len=args.max_len); opt = torch.optim.AdamW(model.parameters(), lr=args.lr); args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"; best = args.out / "best.pt"; source = latest if latest.exists() else args.base; state = load_checkpoint(source, model=model, optimizer=opt, map_location="cpu"); start = int(state.get("step", 0));
    for g in opt.param_groups: g["lr"] = args.lr
    best_val = float("inf"); gov = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90)); rng = random.Random(20260823 + start); rng.shuffle(train); cursor = 0
    while start < args.steps:
        if cursor >= len(train): rng.shuffle(train); cursor = 0
        item = batch(args.data, [pos[train[cursor]]], tok, args.max_len, pad, bos, eos); cursor += 1
        if item is None: continue
        x, y = item; model.train(); opt.zero_grad(set_to_none=True); logits = model(x, torch.zeros(1,11), tgt_key_padding_mask=(x == pad)); loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); start += 1
        if start % 25 == 0 or start == args.steps:
            val_loss = evaluate(model, args.data, [pos[i] for i in val], tok, args.max_len, pad, bos, eos); metrics={"train_loss":float(loss.detach()),"val_loss":val_loss,"bos":bos,"eos":eos,"learning_rate":args.lr}; save_checkpoint(latest,model=model,optimizer=opt,step=start,metrics=metrics)
            if val_loss < best_val: best_val=val_loss; save_checkpoint(best,model=model,optimizer=opt,step=start,metrics=metrics)
            print(json.dumps({"step":start,**metrics,"best_val":best_val,**gov.throttle()},ensure_ascii=False),flush=True)
    print(f"[OK] Especial concluído; melhor validação {best_val:.5f}",flush=True)


if __name__ == "__main__": main()

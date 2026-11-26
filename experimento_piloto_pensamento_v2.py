"""Piloto isolado e deterministico do decoder textual de pensamento."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from pensamento_linguistico import PensamentoLinguistico


def read_rows(path: Path, limit: int) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("prompt") and row.get("think"):
                    out.append(row)
                if len(out) >= limit:
                    break
    if len(out) < 16:
        raise RuntimeError(f"A fila tem apenas {len(out)} exemplos válidos")
    return out


def encode(tok: Tokenizer, row: dict, max_len: int):
    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    ids = tok.encode(f"{row['prompt']}\n<think>\n{row['think']}").ids
    if bos is not None:
        ids.insert(0, bos)
    if eos is not None:
        ids.append(eos)
    ids = ids[: max_len + 1]
    ids = ids if len(ids) >= 3 else ids + [eos or 0] * (3 - len(ids))
    return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


def batch(tok: Tokenizer, rows: list[dict], max_len: int):
    pairs = [encode(tok, row, max_len) for row in rows]
    width = max(len(x) for x, _ in pairs)
    pad = tok.token_to_id("<pad>") or 0
    xs, ys = [], []
    for x, y in pairs:
        xs.append(F.pad(x, (0, width - len(x)), value=pad))
        ys.append(F.pad(y, (0, width - len(y)), value=-100))
    return torch.stack(xs).long(), torch.stack(ys).long()


def make_model(base: Path, vocab: int, max_len: int):
    model = PensamentoLinguistico(vocab_size=vocab, d_model=384, nhead=6,
                                  num_layers=4, dim_feedforward=1536,
                                  max_len=max_len)
    payload = torch.load(base, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model", payload))
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise RuntimeError(f"Checkpoint incompatível: missing={missing[:2]} extra={extra[:2]}")
    return model


def forward(model, x):
    humor = torch.zeros((x.size(0), 11), dtype=torch.float32)
    return model(x, humor)


@torch.no_grad()
def score(model, x, y):
    model.eval()
    logits = forward(model, x)
    return float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    random.seed(20260827)
    torch.manual_seed(20260827)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data, args.rows)
    cut = max(8, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:cut], rows[cut:]
    tok = Tokenizer.from_file(str(args.tokenizer))
    model = make_model(args.base, tok.get_vocab_size(), args.max_len)
    xval, yval = batch(tok, val_rows, args.max_len)
    before = score(model, xval, yval)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        start = ((step - 1) * 8) % len(train_rows)
        chosen = [train_rows[(start + j) % len(train_rows)] for j in range(min(8, len(train_rows)))]
        x, y = batch(tok, chosen, args.max_len)
        model.train(); opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, x).flatten(0, 1), y.flatten(), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step == args.steps or step % 25 == 0:
            print(f"step={step}/{args.steps} train_loss={loss.item():.6f}", flush=True)
    after = score(model, xval, yval)
    result = {
        "experiment": "pensamento_local_small_pilot_v2",
        "base_checkpoint": str(args.base), "data": str(args.data),
        "rows": len(rows), "train_rows": len(train_rows), "val_rows": len(val_rows),
        "steps": args.steps, "max_len": args.max_len, "lr": args.lr,
        "before_val_loss": before, "after_val_loss": after,
        "delta_after_minus_before": after - before,
        "productive_by_loss": after < before,
        "teacher_approved": False, "production_checkpoint_untouched": True,
    }
    torch.save({"model_state_dict": model.state_dict(), "metrics": result}, args.out / "pilot.pt")
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

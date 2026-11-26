"""Piloto isolado do modulo pensamento.

Este experimento carrega o checkpoint de producao somente para leitura,
treina uma copia por poucos passos e compara a mesma validacao deterministica
antes/depois. A fila usada pode conter exemplos ainda nao aprovados por um
professor remoto; por isso o resultado nunca e promovido automaticamente.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from pensamento_linguistico import PensamentoLinguistico


def load_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            # O piloto treina exclusivamente o rastro privado de pensamento.
            # O campo final fica preservado para auditoria, sem ser misturado
            # no alvo textual deste modulo.
            if row.get("prompt") and row.get("think"):
                rows.append(row)
            if len(rows) >= limit:
                break
    if len(rows) < 16:
        raise RuntimeError(f"Poucos exemplos validos no piloto: {len(rows)}")
    return rows


def encode_example(tok: Tokenizer, row: dict, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    text = f"{row['prompt']}\n<think>\n{row['think']}"
    ids = list(tok.encode(text).ids)
    if bos is not None:
        ids.insert(0, bos)
    if eos is not None:
        ids.append(eos)
    ids = ids[: max_len + 1]
    if len(ids) < 3:
        ids = (ids + [eos if eos is not None else 0] * 3)[:3]
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    return x, y


def batchify(tok: Tokenizer, rows: list[dict], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [encode_example(tok, row, max_len) for row in rows]
    width = max(len(x) for x, _ in pairs)
    pad = tok.token_to_id("<pad>")
    pad = 0 if pad is None else pad
    xs, ys = [], []
    for x, y in pairs:
        n = width - len(x)
        xs.append(F.pad(x, (0, n), value=pad))
        ys.append(F.pad(y, (0, n), value=-100))
    return torch.stack(xs), torch.stack(ys)


def load_model(base: Path, vocab_size: int, max_len: int) -> PensamentoLinguistico:
    model = PensamentoLinguistico(
        vocab_size=vocab_size,
        d_model=384,
        nhead=6,
        num_layers=4,
        dim_feedforward=1536,
        max_len=max_len,
    )
    payload = torch.load(base, map_location="cpu")
    state = payload.get("model_state_dict", payload.get("model", payload))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint incompatível; ausentes={missing[:3]}, extras={unexpected[:3]}")
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
    return float(loss.item())


def main() -> None:
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

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    random.seed(20260827)
    torch.manual_seed(20260827)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data, args.rows)
    split = max(8, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split], rows[split:]
    tok = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.base, tok.get_vocab_size(), args.max_len)
    x_val, y_val = batchify(tok, val_rows, args.max_len)
    before = evaluate(model, x_val, y_val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for step in range(1, args.steps + 1):
        # Amostra deterministica em lotes pequenos para manter o experimento leve.
        start = ((step - 1) * 8) % len(train_rows)
        chosen = [train_rows[(start + j) % len(train_rows)] for j in range(min(8, len(train_rows)))]
        x, y = batchify(tok, chosen, args.max_len)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == args.steps or step % 25 == 0:
            print(f"step={step}/{args.steps} train_loss={loss.item():.6f}", flush=True)

    after = evaluate(model, x_val, y_val)
    delta = after - before
    result = {
        "experiment": "pensamento_local_small_pilot",
        "base_checkpoint": str(args.base),
        "data": str(args.data),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "steps": args.steps,
        "max_len": args.max_len,
        "lr": args.lr,
        "before_val_loss": before,
        "after_val_loss": after,
        "delta_after_minus_before": delta,
        "productive_by_loss": bool(after < before),
        "teacher_approved": False,
        "production_checkpoint_untouched": True,
    }
    torch.save({"model_state_dict": model.state_dict(), "metrics": result}, args.out / "pilot.pt")
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Piloto da Voz intermediária (~50M parâmetros) em linguagem natural limpa."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .checkpoint_manager_safe import save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .treino_voz_texto_limpo import evaluate, offsets, sample
from .voz import Voz


DMODEL = 512
NHEAD = 8
LAYERS = 8
FFN = 2048


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--save-every", type=int, default=25)
    a = p.parse_args()

    random.seed(20260824)
    torch.manual_seed(20260824)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(a.tokenizer))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    positions = offsets(a.data)
    order = list(range(len(positions)))
    random.shuffle(order)
    val = order[:max(1, len(order) // 10)]
    train = order[len(val):]

    model = Voz(
        vocab_size=tok.get_vocab_size(), d_model=DMODEL, nhead=NHEAD,
        num_layers=LAYERS, dim_feedforward=FFN, max_len=a.max_len,
        pad_token_id=pad, bos_token_id=bos, eos_token_id=eos,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    latest, best = a.out / "latest.pt", a.out / "best.pt"
    best_val = float("inf")
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90))
    rng = random.Random(20260824)

    print(json.dumps({
        "event": "model_initialized", "parameters": parameter_count,
        "d_model": DMODEL, "layers": LAYERS, "ffn": FFN,
        "vocab_size": tok.get_vocab_size(), "dataset": "voz_natural_only",
    }, ensure_ascii=False), flush=True)

    for step in range(1, a.steps + 1):
        item = sample(a.data, positions[train[rng.randrange(len(train))]], tok,
                      a.max_len, bos, eos)
        if item is None:
            continue
        x, y = item
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % a.save_every == 0 or step == a.steps:
            val_loss = evaluate(model, a.data, [positions[i] for i in val],
                                tok, a.max_len, bos, eos)
            metrics = {
                "train_loss": float(loss.detach()), "val_loss": val_loss,
                "learning_rate": a.lr, "dataset": "voz_natural_only",
                "vocab_size": tok.get_vocab_size(), "parameters": parameter_count,
                "architecture": {"d_model": DMODEL, "nhead": NHEAD,
                                  "layers": LAYERS, "ffn": FFN},
            }
            save_checkpoint(latest, model=model, optimizer=optimizer,
                            step=step, metrics=metrics)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best, model=model, optimizer=optimizer,
                                step=step, metrics=metrics)
            print(json.dumps({"step": step, **metrics, "best_val": best_val,
                              **governor.throttle()}, ensure_ascii=False), flush=True)

    print(f"[OK] Piloto Voz 50M concluído; melhor validação {best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()

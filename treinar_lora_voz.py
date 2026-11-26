"""Piloto LoRA da Voz 50M com pesos-base congelados."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
    from .resource_governor import ResourceGovernor, ResourcePolicy
    from .treino_voz_texto_limpo import evaluate, offsets, sample
    from .voz import Voz
except (ImportError, ValueError):
    from checkpoint_manager_safe import load_checkpoint, save_checkpoint
    from resource_governor import ResourceGovernor, ResourcePolicy
    from treino_voz_texto_limpo import evaluate, offsets, sample
    from voz import Voz


class LoRALinear(nn.Module):
    """Linear congelada + atualização de baixo posto treinável."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(self.dropout(x), self.lora_a)
        update = F.linear(update, self.lora_b) * self.scaling
        return self.base(x) + update


def add_lora(model: Voz, rank: int, alpha: float, dropout: float) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False

    targets = []
    for layer in model.decoder.layers:
        targets.extend([(layer, "linear1"), (layer, "linear2"),
                        (layer.self_attn, "out_proj"),
                        (layer.multihead_attn, "out_proj")])
    targets.append((model, "lm_head"))

    trainable = 0
    for parent, name in targets:
        original = getattr(parent, name)
        wrapped = LoRALinear(original, rank=rank, alpha=alpha,
                             dropout=dropout)
        setattr(parent, name, wrapped)
        trainable += wrapped.lora_a.numel() + wrapped.lora_b.numel()
    return trainable


def set_lora_train_mode(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.dropout.train()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
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

    model = Voz(vocab_size=tok.get_vocab_size(), d_model=512, nhead=8,
                num_layers=8, dim_feedforward=2048, max_len=a.max_len,
                pad_token_id=pad, bos_token_id=bos, eos_token_id=eos)
    base_state = load_checkpoint(a.base, model=model, map_location="cpu")
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = add_lora(model, a.rank, a.alpha, 0.05)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=a.lr,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    latest, best = a.out / "latest.pt", a.out / "best.pt"
    best_val = float("inf")
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,
                                                ram_limit_percent=90))
    rng = random.Random(20260824)

    print(json.dumps({"event": "lora_initialized",
                      "base_step": base_state.get("step", 0),
                      "total_parameters": total_params,
                      "trainable_parameters": trainable_params,
                      "rank": a.rank, "alpha": a.alpha,
                      "targets": "ffn+attention_out+lm_head"},
                     ensure_ascii=False), flush=True)

    for step in range(1, a.steps + 1):
        item = sample(a.data, positions[train[rng.randrange(len(train))]], tok,
                      a.max_len, bos, eos)
        if item is None:
            continue
        x, y = item
        set_lora_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()

        if step % a.save_every == 0 or step == a.steps:
            val_loss = evaluate(model, a.data, [positions[i] for i in val],
                                tok, a.max_len, bos, eos)
            metrics = {"train_loss": float(loss.detach()), "val_loss": val_loss,
                       "dataset": "voz_natural_only", "total_parameters": total_params,
                       "trainable_parameters": trainable_params, "rank": a.rank,
                       "base": str(a.base)}
            save_checkpoint(latest, model=model, optimizer=optimizer,
                            step=step, metrics=metrics)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best, model=model, optimizer=optimizer,
                                step=step, metrics=metrics)
            print(json.dumps({"step": step, **metrics, "best_val": best_val,
                              **governor.throttle()}, ensure_ascii=False), flush=True)

    print(f"[OK] LoRA Voz concluído; melhor validação {best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()

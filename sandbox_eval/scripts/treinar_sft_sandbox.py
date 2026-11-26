"""Treinamento SFT de LoRA no Sandbox Isolado (Milestone M3).

Treina adaptadores LoRA sobre o decoder de Voz 50M utilizando o dataset
curado pelo Professor Gemini (`sandbox_eval/dados/treino_voz_curado.jsonl`),
sob limites estritos do SandboxResourceGovernor e salvando checkpoints seguros
com estado completo de RNG via checkpoint_manager_safe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

CURRENT_DIR = Path(__file__).resolve().parent
SANDBOX_DIR = CURRENT_DIR.parent
ROOT_DIR = SANDBOX_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import checkpoint_manager_safe
from sandbox_resource_governor import ResourcePolicy, SandboxResourceGovernor
from voz import Voz


# =====================================================================
# Adaptador LoRALinear
# =====================================================================

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(self.dropout(x), self.lora_a)
        update = F.linear(update, self.lora_b) * self.scaling
        return self.base(x) + update


def add_lora(model: Voz, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> int:
    for p in model.parameters():
        p.requires_grad = False

    targets = []
    for layer in model.decoder.layers:
        targets.extend([
            (layer, "linear1"),
            (layer, "linear2"),
            (layer.self_attn, "out_proj"),
            (layer.multihead_attn, "out_proj"),
        ])
    targets.append((model, "lm_head"))

    trainable = 0
    for parent, name in targets:
        original = getattr(parent, name)
        wrapped = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, name, wrapped)
        trainable += wrapped.lora_a.numel() + wrapped.lora_b.numel()
    return trainable


# =====================================================================
# Dataset e Tokenização SFT
# =====================================================================

def carregar_dados_sft(dataset_path: Path, tok: Tokenizer, max_len: int = 128, bos: int = 1, eos: int = 2):
    exemplos = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_ids = tok.encode(item["prompt"], add_special_tokens=False).ids
            response_ids = tok.encode(item["response"], add_special_tokens=False).ids + [eos]
            if not response_ids:
                continue
            response_ids = response_ids[: max_len - 2]
            prompt_budget = max(1, max_len - 1 - len(response_ids))
            prompt_ids = prompt_ids[-prompt_budget:]
            sequence = [bos] + prompt_ids + response_ids
            x = torch.tensor(sequence[:-1], dtype=torch.long)
            labels = torch.tensor(sequence[1:], dtype=torch.long)
            response_start = 1 + len(prompt_ids)
            labels[: max(0, response_start - 1)] = -100

            # Sintetiza vetor de humor realista para condicionamento da cross-attention
            # [L, Q, F, dopamina, serotonina, noradrenalina, adrenalina, gaba, ocitocina, acetilcolina, glutamato]
            humor = torch.tensor([
                random.uniform(0.8, 2.5),
                random.uniform(0.8, 2.5),
                random.uniform(0.2, 1.2),
                random.uniform(0.4, 0.8),
                random.uniform(0.5, 0.9),
                random.uniform(0.2, 0.5),
                random.uniform(0.1, 0.4),
                random.uniform(0.5, 0.8),
                random.uniform(0.5, 0.9),
                random.uniform(0.4, 0.7),
                random.uniform(0.3, 0.6),
            ], dtype=torch.float32)

            exemplos.append((x, labels, humor))
    return exemplos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=20)
    args = p.parse_args()

    random.seed(20261018)
    torch.manual_seed(20261018)
    torch.set_num_threads(2)
    os.environ["HF_HUB_OFFLINE"] = "1"

    governor = SandboxResourceGovernor(
        ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=60.0),
        num_threads=2
    )

    tok_path = ROOT_DIR / "tokenizer_voz_novo_16k.json"
    tok = Tokenizer.from_file(str(tok_path))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2

    # Inicializa arquitetura Voz 50M
    model = Voz(
        vocab_size=tok.get_vocab_size(),
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        max_len=args.max_len,
        pad_token_id=pad,
        bos_token_id=bos,
        eos_token_id=eos
    )

    # Carrega pesos baseline
    add_lora(model, rank=8, alpha=16.0, dropout=0.05)
    base_ckpt = Path("/tmp/best.pt")
    if not base_ckpt.exists():
        base_ckpt = ROOT_DIR / "checkpoints_voz_lora_sft" / "best.pt"

    print(f"[Treino] Carregando pesos baseline de {base_ckpt}...")
    payload = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])

    # Carrega dataset curado
    data_path = SANDBOX_DIR / "dados" / "treino_voz_curado.jsonl"
    print(f"[Treino] Carregando dataset de {data_path}...")
    exemplos = carregar_dados_sft(data_path, tok, max_len=args.max_len, bos=bos, eos=eos)
    random.shuffle(exemplos)

    n_val = max(10, int(len(exemplos) * 0.15))
    val_data = exemplos[:n_val]
    train_data = exemplos[n_val:]
    print(f"[Treino] {len(train_data)} exemplos de treino, {len(val_data)} exemplos de validação.")

    # Otimizador AdamW sobre parâmetros LoRA treináveis
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    out_dir = SANDBOX_DIR / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best.pt"

    best_val_loss = float("inf")
    print(f"[Treino] Iniciando loop SFT por {args.steps} passos...")

    for step in range(1, args.steps + 1):
        model.train()
        x, labels, humor = random.choice(train_data)

        optimizer.zero_grad()
        logits = model(x.unsqueeze(0), humor.unsqueeze(0))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels, ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        # Avaliação periódica e governança
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vlabels, vhumor in val_data:
                    vlogits = model(vx.unsqueeze(0), vhumor.unsqueeze(0))
                    vloss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vlabels, ignore_index=-100)
                    val_losses.append(vloss.item())

            mean_val_loss = sum(val_losses) / len(val_losses)
            gov_status = governor.throttle()

            print(
                f"[Passo {step:3d}/{args.steps}] Train Loss: {loss.item():.4f} | "
                f"Val Loss: {mean_val_loss:.4f} | RAM: {gov_status['ram_percent']:.1f}% | "
                f"CPU: {gov_status['cpu_percent']:.1f}%"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                metrics = {
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": mean_val_loss,
                    "teacher": "Gemini-Flash-High-PhD-Auditor",
                    "dataset": "treino_voz_curado.jsonl"
                }
                checkpoint_manager_safe.save_checkpoint(
                    str(best_ckpt_path),
                    model=model,
                    optimizer=optimizer,
                    epoch=1,
                    step=step,
                    metrics=metrics
                )
                print(f" -> Novo melhor checkpoint salvo em {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    print(f"\n[OK] Treinamento concluído com sucesso. Melhor Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

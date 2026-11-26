"""Treinamento LoRA SFT em Escala Real da Voz do BrotherTrue (Proporção 75/25).

Executa fine-tuning supervisionado dos adaptadores LoRA e cross-attention
sobre a base consolidada (75% Síntese do Professor Gemini + 25% HuggingFace),
iniciando a partir do checkpoint homologado no sandbox (`sandbox_eval/checkpoints/best.pt`).

Garante:
- Operação cooperativa sob HardwareSentry (2 threads CPU, RAM < 60%, piso SSD 4.5 GB)
- Preservação estrita dos geradores de números pseudo-aleatórios (Python, NumPy, PyTorch RNG)
- Salvamento atômico e seguro via checkpoint_manager_safe
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import checkpoint_manager_safe
from hardware_sentry import HardwareSentry, SentryPolicy
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
# Dataset e Tokenização SFT 75/25
# =====================================================================

def carregar_dados_sft(dataset_path: Path, tok: Tokenizer, max_len: int = 128, bos: int = 1, eos: int = 2):
    exemplos = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("prompt", "")
            response = item.get("resposta", item.get("response", ""))
            humor_list = item.get("vetor_humor", [0.1]*11)

            prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
            response_ids = tok.encode(response, add_special_tokens=False).ids + [eos]
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

            humor = torch.tensor(humor_list, dtype=torch.float32)
            exemplos.append((x, labels, humor))
    return exemplos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=15)
    args = p.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    os.environ["HF_HUB_OFFLINE"] = "1"

    sentry = HardwareSentry(
        SentryPolicy(cpu_limit_percent=75.0, ram_limit_percent=60.0, min_disk_free_gb=4.5),
        num_threads=2
    )

    status_inicial = sentry.throttle()
    print(f"[Sentinela] Condições iniciais: RAM={status_inicial['ram_percent']}% | CPU={status_inicial['cpu_percent']}% | SSD Livre={status_inicial['disk_free_gb']} GB")

    tok_path = ROOT_DIR / "tokenizer_voz_novo_16k.json"
    tok = Tokenizer.from_file(str(tok_path))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2

    # Instancia modelo Voz 50M
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

    trainable_count = add_lora(model, rank=8, alpha=16.0, dropout=0.05)
    print(f"[Modelo] LoRA injetado com sucesso: {trainable_count:,} parâmetros treináveis.")

    # Carrega pesos a partir do baseline do sandbox
    base_ckpt = ROOT_DIR / "sandbox_eval" / "checkpoints" / "best.pt"
    if not base_ckpt.exists():
        base_ckpt = ROOT_DIR / "checkpoints_voz_lora_sft" / "best.pt"

    print(f"[Modelo] Carregando pesos pré-treinados de {base_ckpt}...")
    payload = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])

    # Carrega dados 75/25
    train_path = ROOT_DIR / "treino_real_escala" / "dados" / "treino_real_consolidado.jsonl"
    val_path = ROOT_DIR / "treino_real_escala" / "dados" / "val_real_consolidado.jsonl"

    train_data = carregar_dados_sft(train_path, tok, max_len=args.max_len, bos=bos, eos=eos)
    val_data = carregar_dados_sft(val_path, tok, max_len=args.max_len, bos=bos, eos=eos)
    print(f"[Dados 75/25] {len(train_data)} amostras de treino, {len(val_data)} amostras de validação.")

    # Otimizador AdamW
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    out_dir = ROOT_DIR / "treino_real_escala" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT_DIR / "treino_real_escala" / "relatorios"
    report_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(split_data):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for x, labels, humor in split_data:
                x_in = x.unsqueeze(0)
                lbl = labels.unsqueeze(0)
                hum_in = humor.unsqueeze(0)
                logits = model(x_in, humor=hum_in)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl.view(-1), ignore_index=-100)
                valid = (lbl != -100).sum().item()
                if valid > 0:
                    total_loss += loss.item() * valid
                    total_tokens += valid
        return total_loss / max(1, total_tokens)

    # Avaliação Baseline
    baseline_val_loss = evaluate(val_data)
    print(f"[Baseline 75/25] Loss Inicial de Validação: {baseline_val_loss:.4f}")

    best_val_loss = baseline_val_loss
    history = []
    start_time = time.time()

    print(f"[Treino] Iniciando loop de otimização ({args.steps} passos)...")
    for step in range(1, args.steps + 1):
        # Proteção e sentinela
        sent_info = sentry.throttle()

        model.train()
        x, labels, humor = random.choice(train_data)
        x_in = x.unsqueeze(0)
        lbl = labels.unsqueeze(0)
        hum_in = humor.unsqueeze(0)

        optimizer.zero_grad()
        logits = model(x_in, humor=hum_in)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(val_data)
            elapsed = time.time() - start_time
            print(
                f"Passo {step:03d}/{args.steps:03d} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | RAM: {sent_info['ram_percent']}% | "
                f"CPU: {sent_info['cpu_percent']}% | Disco: {sent_info['disk_free_gb']} GB"
            )

            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "ram_percent": sent_info["ram_percent"],
                "cpu_percent": sent_info["cpu_percent"],
                "disk_free_gb": sent_info["disk_free_gb"],
                "elapsed_seconds": round(elapsed, 1),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = out_dir / "best.pt"
                checkpoint_manager_safe.save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=1,
                    step=step,
                    metrics={"val_loss": val_loss, "train_loss": loss.item()}
                )
                print(f"⭐ [Melhor Checkpoint] Salvo atomicamente em {best_path} (Val Loss: {val_loss:.4f})")

    # Salva checkpoint final
    final_path = out_dir / "checkpoint_final.pt"
    checkpoint_manager_safe.save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        step=args.steps,
        metrics={"final_val_loss": val_loss, "best_val_loss": best_val_loss}
    )

    # Salva telemetria completa
    telemetria = {
        "dataset": "treino_real_escala_consolidado_75_25",
        "steps": args.steps,
        "lr": args.lr,
        "baseline_val_loss": round(baseline_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "loss_reduction_pct": round(100.0 * (baseline_val_loss - best_val_loss) / baseline_val_loss, 2),
        "total_elapsed_seconds": round(time.time() - start_time, 1),
        "history": history,
    }

    telemetria_path = report_dir / "telemetria_treino_real.json"
    with open(telemetria_path, "w", encoding="utf-8") as f:
        json.dump(telemetria, f, indent=2, ensure_ascii=False)

    print(f"[OK] Treinamento concluído com sucesso!")
    print(f"📊 Telemetria registrada em: {telemetria_path}")
    print(f"🎯 Redução de Val Loss: {baseline_val_loss:.4f} -> {best_val_loss:.4f} ({telemetria['loss_reduction_pct']}%)")


if __name__ == "__main__":
    main()

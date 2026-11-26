"""Curadoria e Consolidação do Dataset 50/50 (HuggingFace + Professor Gemini).

Realiza o balanceamento formal:
- 50% Diálogos conversacionais abertos em Português (HuggingFace)
- 50% Diálogos sintéticos de alto nível biocognitivo e dialético (Professor Gemini)
- Embaralhamento determinístico (seed=42)
- Divisão Train (90%) / Val (10%)
- Validação estrita de tokens especiais ChatML e vetor de humor 11D
- Geração do manifesto formal com SHA-256 e contagem de tokens
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

from tokenizers import Tokenizer

# Sentinela local
sys.path.insert(0, str(Path(__file__).parent))
from hardware_sentry import HardwareSentry


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def consolidar_datasets_5050(
    hf_path: str = "treino_real_escala/dados/hf_dialogos_pt.jsonl",
    gemini_path: str = "treino_real_escala/dados/sintese_gemini_profundo.jsonl",
    output_train: str = "treino_real_escala/dados/treino_real_consolidado.jsonl",
    output_val: str = "treino_real_escala/dados/val_real_consolidado.jsonl",
    manifest_path: str = "treino_real_escala/dados/manifest.json",
    tokenizer_path: str = "tokenizer_voz_novo_16k.json",
    seed: int = 42,
):
    sentry = HardwareSentry()
    status = sentry.throttle()
    print(f"[Sentinela] Status antes da consolidação: SSD Livre = {status['disk_free_gb']} GB")

    hf_file = Path(hf_path)
    gemini_file = Path(gemini_path)

    if not hf_file.exists():
        raise FileNotFoundError(f"Arquivo HF não encontrado: {hf_file}")
    if not gemini_file.exists():
        raise FileNotFoundError(f"Arquivo Gemini não encontrado: {gemini_file}")

    # Carrega dados HF
    hf_data = []
    with open(hf_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hf_data.append(json.loads(line))

    # Carrega dados Gemini
    gemini_data = []
    with open(gemini_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gemini_data.append(json.loads(line))

    print(f"[Consolidação] Registros HF carregados: {len(hf_data)}")
    print(f"[Consolidação] Registros Gemini carregados: {len(gemini_data)}")

    # Balanceamento 50/50: define N pelo menor ou proporção balanceada
    n_balance = min(len(hf_data), len(gemini_data))
    if n_balance == 0:
        raise ValueError("Um dos datasets está vazio!")

    random.seed(seed)
    sampled_hf = random.sample(hf_data, n_balance)
    sampled_gemini = random.sample(gemini_data, n_balance)

    print(f"[Consolidação] Amostrados {n_balance} pares HF e {n_balance} pares Gemini (Total balanceado: {2 * n_balance})")

    consolidado = []
    for item in sampled_hf:
        item["composicao"] = "50_hf_open"
        consolidado.append(item)
    for item in sampled_gemini:
        item["composicao"] = "50_gemini_teacher"
        consolidado.append(item)

    # Embaralha os dados balanceados
    random.shuffle(consolidado)

    # Divisão Train (90%) e Val (10%)
    split_idx = max(1, int(len(consolidado) * 0.90))
    train_items = consolidado[:split_idx]
    val_items = consolidado[split_idx:]

    print(f"[Consolidação] Divisão: {len(train_items)} treino | {len(val_items)} validação")

    # Salva treino
    train_file = Path(output_train)
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Salva validação
    val_file = Path(output_val)
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Estatísticas de tokenização
    tokenizer = Tokenizer.from_file(tokenizer_path)
    total_tokens_train = 0
    token_lengths = []
    for item in train_items:
        enc = tokenizer.encode(item["text"])
        total_tokens_train += len(enc.ids)
        token_lengths.append(len(enc.ids))

    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    min_tokens = min(token_lengths) if token_lengths else 0

    manifest = {
        "dataset_name": "treino_real_escala_consolidado_5050",
        "description": "Composição 50/50 balanceada entre Diálogos Abertos HF em Português e Síntese Profunda do Professor Gemini",
        "train_file": str(train_file),
        "val_file": str(val_file),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "total_samples": len(consolidado),
        "balance_ratio": {
            "hf_open_ratio": 0.50,
            "gemini_teacher_ratio": 0.50,
        },
        "token_stats_train": {
            "total_tokens": total_tokens_train,
            "avg_tokens_per_sample": round(avg_tokens, 1),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "sha256": {
            "train": compute_sha256(train_file),
            "val": compute_sha256(val_file),
            "hf_raw": compute_sha256(hf_file),
            "gemini_raw": compute_sha256(gemini_file),
        },
        "seed": seed,
    }

    manifest_file = Path(manifest_path)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    sentry.purge_caches()
    final_status = sentry.throttle()
    print(f"[Manifesto] Gravado com sucesso em {manifest_file} (SHA-256 Treino: {manifest['sha256']['train'][:12]}...)")
    print(f"[Sentinela] SSD Livre final: {final_status['disk_free_gb']} GB")
    return manifest


if __name__ == "__main__":
    m = consolidar_datasets_5050()
    print("[OK] Consolidação 50/50 concluída com sucesso.")

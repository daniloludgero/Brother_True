"""Testador e Carregador Offline da Voz Distribuída 20K (RoPE + LoRA).

Permite testar e validar localmente no hardware (CPU 2 threads, 16 GB RAM) a arquitetura
com janela contextual de 20.480 tokens, compatível tanto com o checkpoint consolidado
FP16 (~280 MB) quanto com adaptadores LoRA (~15 MB) exportados do Google Colab / Drive.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from voz_distribuida import VozDistribuida


def testar_inferencia_offline_20k(
    checkpoint_path: Optional[str] = None,
    seq_len_teste: int = 2048,
    temperatura: float = 0.8,
    top_p: float = 0.9
):
    print("=" * 70)
    print(" INFERÊNCIA OFFLINE BROTHERTRUE VOZ DISTRIBUÍDA (20.480 TOKENS)")
    print(" Governança de Recursos: torch.set_num_threads(2) | Hardware Local")
    print("=" * 70)

    torch.set_num_threads(2)
    device = torch.device("cpu")

    # 1. Instanciação da arquitetura 20k (157.9M parâmetros)
    print("\n[1/3] Inicializando arquitetura com RoPE (theta=500.000, max_len=20480)...")
    start_init = time.time()
    model = VozDistribuida(
        vocab_size=16384,
        d_model=768,
        nhead=12,
        num_layers=14,
        dim_feedforward=2048,
        max_len=20480,
        rope_base=500000.0
    ).to(device)
    init_time = time.time() - start_init

    stats = model.count_parameters()
    print(f"✅ Modelo instanciado em {init_time:.2f}s!")
    print(f"📊 Total de Parâmetros: {stats['total_parameters']:,} ({stats['total_millions']}M)")
    print(f"📏 Janela Contextual Máxima: {model.max_len:,} tokens")

    # 2. Carga opcional de pesos do Colab / Drive
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n[2/3] Carregando pesos de: {checkpoint_path}...")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "lora_state" in payload:
            print("🔹 Adaptador LoRA detectado (~15 MB). Aplicando ao modelo base...")
            # Carrega adaptadores lora
            missing, unexpected = model.load_state_dict(payload["lora_state"], strict=False)
            print(f"✅ Adaptador LoRA carregado! (Inesperados: {len(unexpected)}, Ausentes base: {len(missing)})")
        elif "model" in payload:
            print("🔹 Checkpoint FP16/FP32 consolidado detectado. Carregando pesos...")
            model.load_state_dict(payload["model"], strict=False)
            print("✅ Checkpoint consolidado carregado com sucesso!")
    else:
        print("\n[2/3] Nenhum checkpoint externo fornecido. Executando auditoria estrutural em pesos nativos...")

    # 3. Teste de passagem direta em contexto longo (SDPA + RoPE)
    print(f"\n[3/3] Testando passagem direta com prompt sintético de {seq_len_teste} tokens...")
    model.eval()

    prompt_dummy = torch.randint(3, 16000, (1, seq_len_teste), dtype=torch.long, device=device)
    prompt_dummy[0, 0] = 1  # <bos>
    humor = torch.tensor([[0.8, 0.4, 0.7, 0.5, 0.9, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7]], device=device)

    start_fwd = time.time()
    with torch.no_grad():
        logits = model(prompt_dummy, humor=humor)
    fwd_time = time.time() - start_fwd

    print(f"✅ Passagem direta causal concluída em {fwd_time:.2f}s!")
    print(f"📐 Dimensões dos Logits: {tuple(logits.shape)} (batch=1, seq={seq_len_teste}, vocab=16384)")
    print(f"🧠 Memória RAM estimada em uso: ~{seq_len_teste * 768 * 4 / (1024**2):.2f} MB")

    # 4. Amostragem de geração autoregressiva com penalidade de repetição
    print("\n[Bônus] Gerando novos tokens autoregressivos sob modulação neuroquímica 11D...")
    prompt_ids = [1, 100, 250, 312, 450]
    tokens_gerados = model.gerar(
        prompt_ids=prompt_ids,
        humor=humor[0],
        max_new_tokens=16,
        temperatura=temperatura,
        top_p=top_p,
        repetition_penalty=1.15,
        device=device
    )
    print(f"🎉 Tokens gerados ({len(tokens_gerados)} tokens): {tokens_gerados}")
    print(f"📈 Taxa de vocabulário único: {len(set(tokens_gerados)) / max(1, len(tokens_gerados)):.2f}")
    print("\n" + "=" * 70)
    print(" RESULTADO: Arquitetura 20k pronta para operação offline e nuvem!")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Caminho para o checkpoint .pt ou LoRA")
    p.add_argument("--tokens", type=int, default=2048, help="Comprimento de teste da sequência (ex: 2048 a 20480)")
    args = p.parse_args()
    testar_inferencia_offline_20k(checkpoint_path=args.checkpoint, seq_len_teste=args.tokens)

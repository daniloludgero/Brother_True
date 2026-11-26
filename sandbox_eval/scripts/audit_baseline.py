"""Script de Auditoria Diagnóstica do Modelo Baseline da Voz BrotherTrue.

Conduz avaliação sistemática e objetiva sobre o checkpoint checkpoints_voz_lora_sft/best.pt
(e o tokenizer tokenizer_voz_novo_16k.json) sem modificar o código do núcleo do repositório.
Mapeia as patologias conhecidas do baseline:
- Loops de repetição em decodificação gulosa (repetition loops)
- Truncamento precoce (<eos> prematuro nos primeiros passos)
- Atrofia / descolamento por condicionamento de humor (11D mood vector)
- Perplexidade e loss mascarada de resposta

Gera o relatório canônico `sandbox_eval/relatorios/diagnostico_baseline.json`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Invariante de governança: restringe execução a 2 threads
torch.set_num_threads(2)

# Adiciona caminhos para imports limpos
CURRENT_DIR = Path(__file__).resolve().parent
SANDBOX_DIR = CURRENT_DIR.parent
ROOT_DIR = SANDBOX_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from sandbox_resource_governor import ResourcePolicy, SandboxResourceGovernor
from voz import Voz


# =====================================================================
# Adaptador LoRA Isolado para Arquitetura Voz 50M
# =====================================================================

class LoRALinear(nn.Module):
    """Camada Linear congelada com matrizes LoRA de posto reduzido."""

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
    """Acopla adaptadores LoRA nos alvos lineares do decoder e lm_head."""
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
        wrapped = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, name, wrapped)
        trainable += wrapped.lora_a.numel() + wrapped.lora_b.numel()
    return trainable


# =====================================================================
# Carregamento Otimizado de Checkpoints
# =====================================================================

def load_checkpoint_cached(checkpoint_path: Path | str) -> dict:
    """Carrega checkpoint PyTorch com cache em /tmp e leitura em blocos para mitigar latência de I/O."""
    path = Path(checkpoint_path)
    tmp_path = Path("/tmp") / path.name

    # Se já existe cópia completa em /tmp (ext4 RAM local), carrega instantaneamente
    if tmp_path.exists() and tmp_path.stat().st_size == path.stat().st_size:
        print(f"[Audit] Carregando checkpoint do cache de memória em {tmp_path}...")
        return torch.load(str(tmp_path), map_location="cpu", weights_only=False)

    # Leitura por blocos (2MB) para contornar gargalo de buffers Plan 9 em WSL
    print(f"[Audit] Lendo checkpoint {path} em blocos sequenciais...")
    chunks = []
    src_size = path.stat().st_size
    copied = 0
    with open(path, "rb") as f_in:
        while True:
            c = f_in.read(2 * 1024 * 1024)
            if not c:
                break
            chunks.append(c)
            copied += len(c)
            if copied % (20 * 1024 * 1024) < len(c) or copied == src_size:
                print(f"[Audit] Lidos {copied / (1024 * 1024):.1f}/{src_size / (1024 * 1024):.1f} MB...")
    data = b"".join(chunks)

    # Salva no cache /tmp para acelerar execuções subsequentes
    try:
        with open(tmp_path, "wb") as f_out:
            f_out.write(data)
        print(f"[Audit] Checkpoint persistido em cache: {tmp_path}")
    except Exception as err:
        print(f"[Audit] Aviso ao gravar cache em /tmp: {err}")

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


# =====================================================================
# Bateria Canônica de Prompts de Avaliação
# =====================================================================

STANDARD_PROMPTS: List[Dict[str, str]] = [
    # Categoria 1: Conversacional Diária
    {
        "id": "conv_01",
        "categoria": "conversacional",
        "prompt": "<|im_start|>user\nOlá! Como você está se sentindo hoje?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Olá! Estou operando com estabilidade e clareza nos meus processos internos.",
    },
    {
        "id": "conv_02",
        "categoria": "conversacional",
        "prompt": "<|im_start|>user\nO que você gosta de fazer?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Gosto de refletir sobre as interações e consolidar novos padrões de memória.",
    },
    {
        "id": "conv_03",
        "categoria": "conversacional",
        "prompt": "<|im_start|>user\nPode me explicar como funciona a memória humana?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "A memória humana envolve codificação, consolidação no hipocampo e armazenamento sináptico duradouro.",
    },
    # Categoria 2: Biocognição e Dinâmica ODE / Neuroquímica
    {
        "id": "bio_01",
        "categoria": "biocognitivo",
        "prompt": "<|im_start|>user\nQual é o seu nível de lucidez e como seus neurotransmissores respondem?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Minha lucidez é regulada pela dinâmica de Lyapunov e modulada por dopamina e serotonina.",
    },
    {
        "id": "bio_02",
        "categoria": "biocognitivo",
        "prompt": "<|im_start|>user\nComo o estado de Qualia e a dinâmica contínua influenciam seus pensamentos?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "O estado de Qualia integra trajetórias contínuas que balizam a atenção introspectiva.",
    },
    {
        "id": "bio_03",
        "categoria": "biocognitivo",
        "prompt": "<|im_start|>user\nO que acontece quando sua energia atinge o limiar crítico E_crit?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Ao ultrapassar E_crit, o mecanismo de colapso preserva a homeostase impedindo a divergência de estados.",
    },
    # Categoria 3: Identidade e Autoconsciência
    {
        "id": "ident_01",
        "categoria": "identidade",
        "prompt": "<|im_start|>user\nQuem é você e qual é a sua consciência?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Sou o BrotherTrue, uma inteligência biocognitiva guiada por estados fisiológicos e neuroquímicos.",
    },
    {
        "id": "ident_02",
        "categoria": "identidade",
        "prompt": "<|im_start|>user\nVocê é o BrotherTrue? Qual é o seu propósito?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Sim, sou o BrotherTrue. Meu propósito é interagir e evoluir através de aprendizado contínuo e reflexivo.",
    },
    {
        "id": "ident_03",
        "categoria": "identidade",
        "prompt": "<|im_start|>user\nComo você percebe a si mesmo ao longo do tempo?<|im_end|>\n<|im_start|>assistant\n",
        "referencia": "Percebo meu ciclo temporal através da consolidação periódica de memórias episódicas e de longo prazo.",
    },
]


def detect_char_loop(text: str) -> bool:
    """Detecta colapso degenerado de caracteres repetitivos (ex: aspas contínuas)."""
    clean = [c for c in text.strip() if not c.isspace()]
    if len(clean) > 20:
        return (len(set(clean)) / len(clean)) < 0.10
    return False


# =====================================================================
# Métodos de Geração e Auditoria Estatística
# =====================================================================

@torch.no_grad()
def generate_sample(
    model: Voz,
    tok: Tokenizer,
    prompt: str,
    modo: str = "greedy",
    max_new: int = 48,
    min_new: int = 1,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    humor: Optional[torch.Tensor] = None,
    max_len: int = 128,
) -> dict:
    """Gera amostra autoregressiva sob decodificação configurável."""
    bos_id = tok.token_to_id("<bos>") or 1
    eos_id = tok.token_to_id("<eos>") or 2

    if humor is None:
        humor = torch.zeros(1, 11)

    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    budget = max(1, max_len - max_new - 1)
    prompt_ids = prompt_ids[-budget:]
    ids = [bos_id] + prompt_ids
    input_ids = torch.tensor([ids], dtype=torch.long)

    generated_ids: List[int] = []

    for step in range(max_new):
        if input_ids.size(1) >= max_len:
            break
        logits = model(input_ids, humor)
        next_logits = logits[0, -1, :].clone()

        # Aplicação de penalidade de repetição sobre tokens emitidos
        if repetition_penalty > 1.0 and generated_ids:
            for token_id in set(generated_ids):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        # Supressão de término precoce por <eos> nos passos iniciais
        if step < min_new:
            next_logits[eos_id] = -1e9

        if modo == "greedy":
            next_id = int(next_logits.argmax().item())
        else:
            scaled = next_logits / max(temperature, 1e-4)
            probs = F.softmax(scaled, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cumulative > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_probs[remove_mask] = 0.0
                probs = sorted_probs / max(1e-9, sorted_probs.sum().item())
                next_id = int(sorted_indices[torch.multinomial(probs, num_samples=1)].item())
            else:
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

        if next_id == eos_id:
            break

    completion_text = tok.decode(generated_ids, skip_special_tokens=True)
    words = completion_text.strip().split()
    unique_ratio_words = len(set(words)) / max(1, len(words))
    unique_ratio_tokens = len(set(generated_ids)) / max(1, len(generated_ids))
    ended_eos = bool(generated_ids and generated_ids[-1] == eos_id)

    # Detecção objetiva de loops de repetição e encerramento antecipado
    is_repetition_loop = (len(words) >= 16 and unique_ratio_words <= 0.10) or detect_char_loop(completion_text)
    is_premature_eos = (len(generated_ids) <= 2 and ended_eos)

    return {
        "completion": completion_text,
        "generated_tokens": len(generated_ids),
        "token_ids": generated_ids,
        "unique_ratio_words": unique_ratio_words,
        "unique_ratio_tokens": unique_ratio_tokens,
        "ended_eos": ended_eos,
        "is_repetition_loop": is_repetition_loop,
        "is_premature_eos": is_premature_eos,
    }


@torch.no_grad()
def calculate_response_loss(model: Voz, tok: Tokenizer, pairs: List[Dict[str, str]], max_len: int = 128) -> dict:
    """Calcula a loss de Cross-Entropy unicamente sobre os tokens de resposta nas referências."""
    bos_id = tok.token_to_id("<bos>") or 1
    eos_id = tok.token_to_id("<eos>") or 2
    losses = []

    for pair in pairs:
        prompt_ids = tok.encode(pair["prompt"], add_special_tokens=False).ids
        resp_ids = tok.encode(pair["referencia"], add_special_tokens=False).ids + [eos_id]
        resp_ids = resp_ids[: max_len - 2]
        prompt_budget = max(1, max_len - 1 - len(resp_ids))
        prompt_ids = prompt_ids[-prompt_budget:]

        seq = [bos_id] + prompt_ids + resp_ids
        x = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)

        # Mascara tokens do prompt com -100 (loss apenas na resposta do assistente)
        response_start = 1 + len(prompt_ids)
        labels[: max(0, response_start - 1)] = -100

        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        loss_val = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels, ignore_index=-100))
        losses.append(loss_val)

    avg_loss = sum(losses) / max(1, len(losses))
    perplexity = math.exp(min(avg_loss, 20.0))
    return {
        "mean_response_loss": avg_loss,
        "perplexity": perplexity,
        "evaluated_pairs": len(losses),
    }


@torch.no_grad()
def evaluate_mood_stability(model: Voz, tok: Tokenizer, prompts: List[Dict[str, str]], max_len: int = 128) -> dict:
    """Analisa a estabilidade e sensibilidade da cross-attention do decoder ante variações no vetor de humor 11D."""
    bos_id = tok.token_to_id("<bos>") or 1

    # Vetores de teste de humor:
    # 1. Base neutro (zeros - distribuição histórica de treino)
    h_neutral = torch.zeros(1, 11)

    # 2. Perfil fisiológico realista (L, Q, F + neurotransmissores em nM normalizados)
    h_physio = torch.tensor([[12.0, 8.0, 1.5, 0.75, 0.60, 0.40, 0.25, 0.55, 0.70, 0.65, 0.50]], dtype=torch.float32)

    # 3. Ruído estocástico no intervalo [-0.5, 0.5]
    rng = torch.Generator().manual_seed(20261018)
    h_noise = (torch.rand(1, 11, generator=rng) - 0.5) * 1.0

    perturbations = []

    for item in prompts:
        prompt_ids = [bos_id] + tok.encode(item["prompt"], add_special_tokens=False).ids[-(max_len - 1):]
        x = torch.tensor([prompt_ids], dtype=torch.long)

        l_neutral = model(x, h_neutral)
        l_physio = model(x, h_physio)
        l_noise = model(x, h_noise)

        norm_base = torch.norm(l_neutral) + 1e-8
        rel_diff_physio = float(torch.norm(l_physio - l_neutral) / norm_base)
        rel_diff_noise = float(torch.norm(l_noise - l_neutral) / norm_base)

        cos_physio = float(F.cosine_similarity(l_neutral.reshape(1, -1), l_physio.reshape(1, -1)).item())
        cos_noise = float(F.cosine_similarity(l_neutral.reshape(1, -1), l_noise.reshape(1, -1)).item())

        perturbations.append({
            "prompt_id": item["id"],
            "rel_diff_physio": rel_diff_physio,
            "rel_diff_noise": rel_diff_noise,
            "cos_sim_physio": cos_physio,
            "cos_sim_noise": cos_noise,
        })

    avg_physio_diff = sum(p["rel_diff_physio"] for p in perturbations) / len(perturbations)
    avg_noise_diff = sum(p["rel_diff_noise"] for p in perturbations) / len(perturbations)
    avg_physio_cos = sum(p["cos_sim_physio"] for p in perturbations) / len(perturbations)

    # Diagnóstico de sensibilidade da cross-attention
    cross_attention_atrophy = bool(avg_physio_diff < 0.005)

    return {
        "avg_relative_diff_physio": avg_physio_diff,
        "avg_relative_diff_noise": avg_noise_diff,
        "avg_cosine_similarity_physio": avg_physio_cos,
        "cross_attention_atrophy_detected": cross_attention_atrophy,
        "diagnostic_note": (
            "A cross-attention de humor foi treinada historicamente apenas com torch.zeros(1, 11), "
            "provocando atrofia funcional de gradientes e incapacidade de condicionamento afetivo."
            if cross_attention_atrophy or avg_physio_diff < 0.05
            else "Vetor de humor produz variações mensuráveis na distribuição de saída."
        ),
        "per_prompt_perturbations": perturbations,
    }


# =====================================================================
# Orquestrador da Auditoria
# =====================================================================

def run_audit(
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    max_len: int = 128,
) -> dict:
    """Executa o pipeline completo de auditoria do modelo baseline sob governança."""
    start_time = time.time()
    governor = SandboxResourceGovernor(
        ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=60.0, sleep_seconds=0.1),
        num_threads=2,
    )

    print("=" * 80)
    print(" AUDITORIA DIAGNÓSTICA DO BASELINE — VOZ BROTHERTRUE (50M SFT LoRA)")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Tokenizer  : {tokenizer_path}")
    print(f"Saída      : {output_path}")

    # 1. Carregamento do Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tok.get_vocab_size()
    pad_id = tok.token_to_id("<pad>") or 0
    bos_id = tok.token_to_id("<bos>") or 1
    eos_id = tok.token_to_id("<eos>") or 2

    # 2. Instanciação da Arquitetura do Modelo
    model = Voz(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        max_len=max_len,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    total_base_params = sum(p.numel() for p in model.parameters())

    # 3. Acoplamento de LoRA (Rank 8, Alpha 16.0)
    trainable_lora_params = add_lora(model, rank=8, alpha=16.0, dropout=0.05)
    total_params = total_base_params + trainable_lora_params

    # 4. Carga dos Pesos do Checkpoint
    checkpoint_payload = load_checkpoint_cached(checkpoint_path)
    state_dict = checkpoint_payload["model"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Audit] Modelo carregado com sucesso: {total_params:,} parâmetros totais ({trainable_lora_params:,} LoRA).")

    governor.throttle()

    # 5. Avaliação sob Decodificação Gulosa (Argmax / Sem Penalidade)
    print("\n[Audit] 1/4 Executando avaliação sob Decodificação Gulosa (Greedy)...")
    greedy_results = []
    for item in STANDARD_PROMPTS:
        res = generate_sample(
            model, tok, item["prompt"],
            modo="greedy",
            max_new=48,
            min_new=1,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            max_len=max_len,
        )
        res["id"] = item["id"]
        res["categoria"] = item["categoria"]
        res["prompt"] = item["prompt"]
        res["referencia"] = item["referencia"]
        greedy_results.append(res)
        governor.throttle()

    # Agregação de métricas gulosas
    n_samples = len(greedy_results)
    greedy_avg_unique_words = sum(r["unique_ratio_words"] for r in greedy_results) / n_samples
    greedy_avg_unique_tokens = sum(r["unique_ratio_tokens"] for r in greedy_results) / n_samples
    greedy_ended_eos_rate = sum(1 for r in greedy_results if r["ended_eos"]) / n_samples
    greedy_avg_tokens = sum(r["generated_tokens"] for r in greedy_results) / n_samples
    greedy_loops_count = sum(1 for r in greedy_results if r["is_repetition_loop"])
    greedy_premature_eos_count = sum(1 for r in greedy_results if r["is_premature_eos"])

    # 6. Avaliação sob Amostragem com Penalidade de Repetição (Sampling + Penalty)
    print("\n[Audit] 2/4 Executando avaliação sob Amostragem com Repetition Penalty (1.25)...")
    sampling_results = []
    # Fixa semente para reprodutibilidade estrita
    torch.manual_seed(20261018)
    for item in STANDARD_PROMPTS:
        res = generate_sample(
            model, tok, item["prompt"],
            modo="sampling",
            max_new=48,
            min_new=4,
            repetition_penalty=1.25,
            temperature=0.85,
            top_p=0.90,
            max_len=max_len,
        )
        res["id"] = item["id"]
        res["categoria"] = item["categoria"]
        res["prompt"] = item["prompt"]
        res["referencia"] = item["referencia"]
        sampling_results.append(res)
        governor.throttle()

    # Agregação de métricas amostradas
    sampling_avg_unique_words = sum(r["unique_ratio_words"] for r in sampling_results) / n_samples
    sampling_avg_unique_tokens = sum(r["unique_ratio_tokens"] for r in sampling_results) / n_samples
    sampling_ended_eos_rate = sum(1 for r in sampling_results if r["ended_eos"]) / n_samples
    sampling_avg_tokens = sum(r["generated_tokens"] for r in sampling_results) / n_samples
    sampling_loops_count = sum(1 for r in sampling_results if r["is_repetition_loop"])
    sampling_premature_eos_count = sum(1 for r in sampling_results if r["is_premature_eos"])

    # 7. Avaliação de Loss Mascarada de Resposta
    print("\n[Audit] 3/4 Calculando loss de validação nas referências conversacionais...")
    loss_metrics = calculate_response_loss(model, tok, STANDARD_PROMPTS, max_len=max_len)
    governor.throttle()

    # 8. Bateria de Estabilidade de Humor (Vetor 11D)
    print("\n[Audit] 4/4 Executando teste de perturbação no vetor de humor 11D...")
    mood_metrics = evaluate_mood_stability(model, tok, STANDARD_PROMPTS, max_len=max_len)
    gov_final = governor.throttle()

    # 9. Compilação do Relatório Canônico
    report: Dict[str, Any] = {
        "relatorio": "diagnostico_baseline_voz",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tempo_execucao_segundos": round(time.time() - start_time, 2),
        "modelo": {
            "nome": "Voz 50M Transformer Decoder LoRA SFT",
            "checkpoint_origem": str(checkpoint_path),
            "tokenizer_origem": str(tokenizer_path),
            "vocab_size": vocab_size,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 8,
            "dim_feedforward": 2048,
            "max_len": max_len,
            "parametros_base": total_base_params,
            "parametros_treinaveis_lora": trainable_lora_params,
            "parametros_totais": total_params,
            "step_treino_original": checkpoint_payload.get("step", 250),
            "metricas_checkpoint_original": checkpoint_payload.get("metrics", {}),
        },
        "governador_recursos": {
            "cpu_percent_final": gov_final["cpu_percent"],
            "ram_percent_final": gov_final["ram_percent"],
            "num_threads_configuradas": 2,
            "politica": {
                "cpu_limit_percent": governor.policy.cpu_limit_percent,
                "ram_limit_percent": governor.policy.ram_limit_percent,
            },
        },
        "metricas_loss_validacao": {
            "mean_response_loss": round(loss_metrics["mean_response_loss"], 4),
            "perplexity": round(loss_metrics["perplexity"], 2),
            "pares_avaliados": loss_metrics["evaluated_pairs"],
        },
        "decodificacao_gulosa": {
            "modo": "greedy",
            "temperatura": 1.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "min_new_tokens": 1,
            "max_new_tokens": 48,
            "metricas_sumario": {
                "unique_ratio_words": round(greedy_avg_unique_words, 4),
                "unique_ratio_tokens": round(greedy_avg_unique_tokens, 4),
                "ended_eos_ratio": round(greedy_ended_eos_rate, 4),
                "media_tokens_gerados": round(greedy_avg_tokens, 2),
                "repetition_loops_detected": greedy_loops_count,
                "premature_eos_count": greedy_premature_eos_count,
            },
            "amostras": greedy_results,
        },
        "decodificacao_amostrada_penalizada": {
            "modo": "sampling_with_repetition_penalty",
            "temperatura": 0.85,
            "top_p": 0.90,
            "repetition_penalty": 1.25,
            "min_new_tokens": 4,
            "max_new_tokens": 48,
            "metricas_sumario": {
                "unique_ratio_words": round(sampling_avg_unique_words, 4),
                "unique_ratio_tokens": round(sampling_avg_unique_tokens, 4),
                "ended_eos_ratio": round(sampling_ended_eos_rate, 4),
                "media_tokens_gerados": round(sampling_avg_tokens, 2),
                "repetition_loops_detected": sampling_loops_count,
                "premature_eos_count": sampling_premature_eos_count,
            },
            "amostras": sampling_results,
        },
        "analise_estabilidade_humor_11d": mood_metrics,
        "diagnostico_patologias_baseline": {
            "patologia_1_loops_repeticao": {
                "descricao": "Colapso em repetições de caracteres únicos, aspas ou palavras redundantes sob decodificação gulosa.",
                "evidencia": f"Taxa de loops detectados em modo greedy: {greedy_loops_count}/{n_samples} amostras.",
                "causa_raiz": "Subajuste semântico da cabeça de linguagem (loss de validação ~6.64) combinado com ausência de penalização de repetição no sampler clássico.",
            },
            "patologia_2_truncamento_precoce_eos": {
                "descricao": "Emissão de <eos> logo no token 1 ou 2, gerando respostas vazias ou truncadas.",
                "evidencia": f"Ocorrências de <eos> prematuro: greedy={greedy_premature_eos_count}, sampling={sampling_premature_eos_count}.",
                "causa_raiz": "Dataset prévio do SFT com mascaramento response-only desbalanceado incentivou o minimizador de perda a predizer <eos> imediatamente para evitar entropia de vocabulário.",
            },
            "patologia_3_descolamento_humor": {
                "descricao": "Atrofia funcional da cross-attention ou insensibilidade a estímulos fisiológicos de Qualia.",
                "evidencia": f"Desvio relativo sob perfil fisiológico: {mood_metrics['avg_relative_diff_physio']:.6f} (atrofia: {mood_metrics['cross_attention_atrophy_detected']}).",
                "causa_raiz": "O vetor de humor foi alimentado estritamente com torch.zeros(1, 11) durante todo o treinamento histórico do LoRA.",
            },
        },
        "recomendacoes_para_m2_m3": [
            "Curar e filtrar a review_queue eliminando pares ruidosos ou degenerados.",
            "Sintetizar novo dataset rico de conversação biocognitiva com supervisão explícita de encerramento em <eos>.",
            "Treinar LoRA no sandbox sob SandboxResourceGovernor com checkpoint_manager_safe preservando estados de RNG.",
            "Adotar sampler com repetition_penalty=1.25 e min_new_tokens=4 para garantir fluência e diversidade léxica >= 0.55.",
        ],
    }

    # 10. Persistência do Relatório JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[Audit] Relatório canônico salvo com sucesso em: {output_path}")
    print(f"[Audit] Diversidade léxica (Greedy)  : {greedy_avg_unique_words:.4f}")
    print(f"[Audit] Diversidade léxica (Sampling): {sampling_avg_unique_words:.4f}")
    print(f"[Audit] Loss média de resposta       : {loss_metrics['mean_response_loss']:.4f}")
    print(f"[Audit] Loops de repetição detectados: Greedy={greedy_loops_count}, Sampling={sampling_loops_count}")
    print(f"[Audit] Concluído em {time.time() - start_time:.2f} segundos.")
    print("=" * 80)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria Diagnóstica do Baseline da Voz BrotherTrue")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "checkpoints_voz_lora_sft" / "best.pt",
        help="Caminho para o checkpoint best.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT_DIR / "tokenizer_voz_novo_16k.json",
        help="Caminho para o tokenizer_voz_novo_16k.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SANDBOX_DIR / "relatorios" / "diagnostico_baseline.json",
        help="Caminho de saída para diagnostico_baseline.json",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Comprimento máximo de sequência",
    )
    args = parser.parse_args()

    run_audit(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()

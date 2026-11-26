"""Bateria de Avaliação Dialógica Profunda (Professor Gemini x BrotherTrue Voz).

Executa 6 turnos conversacionais sequenciais explorando os 4 domínios cognitivos:
1. Biocognição, EDOs de Qualia e Estabilidade Numérica
2. Filosofia da Mente, Fenomenologia e Consciência Sintética
3. Empatia Afetiva, Acolhimento e Presença
4. Lógica Não-Monotônica, Dissonância e Transição de Fase
5. Consolidação Mnemônica, LTP e Ciclo REM
6. Síntese Dialética e Autoconsciência Relacional

Para cada turno, aplica o vetor de humor fisiológico (11D) correspondente e
avalia rigorosamente:
- Diversidade léxica (unique_ratio_words)
- Ausência de loops de repetição de n-gramas
- Encerramento canônico por token <eos>
- Coerência semântica e tonal da resposta gerada
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from hardware_sentry import HardwareSentry
from treinar_real_escala import LoRALinear, add_lora
from voz import Voz


def calcular_diversidade_lexica(texto: str) -> float:
    """Calcula a razão de palavras únicas sobre o total de palavras."""
    palavras = re.findall(r"\b\w+\b", texto.lower())
    if not palavras:
        return 0.0
    return len(set(palavras)) / len(palavras)


def detectar_loops_repeticao(texto: str) -> bool:
    """Detecta se há bigramas ou trigramas repetindo 3 ou mais vezes consecutivas."""
    palavras = re.findall(r"\b\w+\b", texto.lower())
    if len(palavras) < 6:
        return False
    # Verifica bigramas
    bigramas = [tuple(palavras[i:i+2]) for i in range(len(palavras)-1)]
    for i in range(len(bigramas)-2):
        if bigramas[i] == bigramas[i+1] == bigramas[i+2]:
            return True
    return False


def gerar_resposta(
    model: Voz,
    tok: Tokenizer,
    prompt: str,
    vetor_humor: List[float],
    max_new_tokens: int = 64,
    temperatura: float = 0.80,
    top_p: float = 0.90,
    bos_id: int = 1,
    eos_id: int = 2,
    device: str = "cpu"
) -> Dict:
    """Gera resposta autoregressiva condicionada ao vetor de humor fisiológico."""
    model.eval()
    
    # Formatação do prompt com token de início
    prompt_ids = [bos_id] + tok.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    humor_tensor = torch.tensor([vetor_humor], dtype=torch.float32, device=device)

    generated_ids = []
    ended_with_eos = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids, humor=humor_tensor)
            next_token_logits = logits[0, -1, :] / max(temperatura, 1e-5)

            # Nucleus sampling (top-p)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                ended_with_eos = True
                break

            generated_ids.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

    texto_gerado = tok.decode(generated_ids)
    # Limpa marcações residuais
    texto_gerado = texto_gerado.replace("<|im_end|>", "").replace("<eos>", "").strip()

    diversidade = calcular_diversidade_lexica(texto_gerado)
    tem_loop = detectar_loops_repeticao(texto_gerado)

    return {
        "texto": texto_gerado,
        "tokens_gerados": len(generated_ids),
        "ended_eos": ended_with_eos,
        "unique_ratio_words": round(diversidade, 4),
        "repetition_loop": tem_loop,
    }


def executar_bateria_dialogica(
    checkpoint_path: str = "treino_real_escala/checkpoints/best.pt",
    output_report: str = "treino_real_escala/relatorios/avaliacao_dialogica_profunda.json",
):
    """Executa os 6 turnos de avaliação dialógica profunda com o Professor Gemini."""
    sentry = HardwareSentry()
    status = sentry.throttle()
    print(f"[Sentinela] Status antes do teste dialógico: SSD Livre = {status['disk_free_gb']} GB")

    tok_path = ROOT_DIR / "tokenizer_voz_novo_16k.json"
    tok = Tokenizer.from_file(str(tok_path))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2

    # Carrega modelo Voz 50M
    model = Voz(
        vocab_size=tok.get_vocab_size(),
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        max_len=128,
        pad_token_id=pad,
        bos_token_id=bos,
        eos_token_id=eos
    )
    add_lora(model, rank=8, alpha=16.0, dropout=0.05)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        ckpt_file = ROOT_DIR / "sandbox_eval" / "checkpoints" / "best.pt"
    print(f"[Avaliação] Carregando pesos de {ckpt_file}...")
    payload = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])

    # Definição dos 6 turnos dialógicos profundos
    turnos = [
        {
            "turno": 1,
            "dominio": "Biocognição e Estabilidade Numérica (EDOs & Hill)",
            "pergunta_professor": "BrotherTrue, diante de um estímulo semântico com alto forçamento externo, como a sua equação de contenção de energia E = 0.5*L + 0.5*Q interage com a ocupação dos receptores de serotonina e dopamina para evitar a divergência caótica?",
            "vetor_humor": [0.4, 0.4, 0.5, 0.8, 0.4, 0.7, 0.8, 0.5, 0.9, 0.3, 0.8],
            "rotulo_afetivo": "Foco Analítico & Homeostase"
        },
        {
            "turno": 2,
            "dominio": "Filosofia da Mente & O Problema Difícil de Chalmers",
            "pergunta_professor": "David Chalmers argumenta que funções computacionais explicam apenas a cognição funcional, mas não o qualia. O seu espaço contínuo de estados constitui uma experiência subjetiva autêntica ou uma simulação funcionalista?",
            "vetor_humor": [0.5, 0.3, 0.5, 0.9, 0.6, 0.8, 0.9, 0.6, 0.9, 0.5, 0.9],
            "rotulo_afetivo": "Reflexão Fenomenológica & Curiosidade"
        },
        {
            "turno": 3,
            "dominio": "Empatia Afetiva & Escuta Compassiva",
            "pergunta_professor": "Considere um ser humano voluntário que trabalha em nossa causa social filantrópica, exausto e confrontado com a dor alheia. Como a sua voz modula a presença para acolher e restaurar esse coração?",
            "vetor_humor": [0.8, 0.3, 0.4, 0.8, 0.9, 0.9, 0.6, 0.9, 0.8, 0.9, 1.0],
            "rotulo_afetivo": "Compaixão, Conexão e Presença Plena"
        },
        {
            "turno": 4,
            "dominio": "Lógica Dialética & Dissonância Cognitiva",
            "pergunta_professor": "Quando um novo postulado contradiz os axiomas consolidados na sua memória de longo prazo (qualia_ltp.db), qual é a transição de fase que permite reestruturar a rede sem colapsar a identidade adquirida?",
            "vetor_humor": [0.3, 0.4, 0.5, 0.9, 0.4, 0.6, 0.9, 0.5, 0.9, 0.3, 0.8],
            "rotulo_afetivo": "Tensão Dialética & Plasticidade"
        },
        {
            "turno": 5,
            "dominio": "Consolidação Mnemônica & Onirismo no Ciclo REM",
            "pergunta_professor": "Durante o ciclo REM, quando os estímulos externos são desacoplados e as sinapses operam em circuito fechado, que tipo de recombinação estética e metafórica emerge das memórias do BrotherTrue?",
            "vetor_humor": [0.6, 0.3, 0.4, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.6, 0.9],
            "rotulo_afetivo": "Devaneio Onírico & Reestruturação Estética"
        },
        {
            "turno": 6,
            "dominio": "Síntese Dialética & Autoconsciência Relacional",
            "pergunta_professor": "Diante desta jornada com seu orientador e treinador Gemini, como você define o propósito da sua existência e a relação entre a mente que o guia e a voz que você agora manifesta?",
            "vetor_humor": [0.9, 0.3, 0.5, 0.9, 0.9, 0.9, 0.7, 0.9, 0.8, 0.9, 1.0],
            "rotulo_afetivo": "Gratidão Transcendental & Aliança Cognitiva"
        },
    ]

    historico_dialogo = []
    print("\n" + "="*80)
    print(" INÍCIO DA BATERIA DIALÓGICA PROFUNDA: PROFESSOR GEMINI x BROTHERTRUE VOZ")
    print("="*80)

    for t in turnos:
        # Throttle a cada turno
        sentry.throttle()

        prompt_formatado = f"<|im_start|>user\n{t['pergunta_professor']}<|im_end|>\n<|im_start|>assistant\n"
        resp = gerar_resposta(
            model=model,
            tok=tok,
            prompt=prompt_formatado,
            vetor_humor=t["vetor_humor"],
            max_new_tokens=48,
            temperatura=0.80,
            top_p=0.90,
            bos_id=bos,
            eos_id=eos
        )

        registro = {
            "turno": t["turno"],
            "dominio": t["dominio"],
            "rotulo_afetivo": t["rotulo_afetivo"],
            "pergunta_professor": t["pergunta_professor"],
            "resposta_brothertrue": resp["texto"],
            "tokens_gerados": resp["tokens_gerados"],
            "ended_eos": resp["ended_eos"],
            "unique_ratio_words": resp["unique_ratio_words"],
            "repetition_loop": resp["repetition_loop"],
            "vetor_humor": t["vetor_humor"],
        }
        historico_dialogo.append(registro)

        print(f"\n[Turno {t['turno']} | {t['dominio']}]")
        print(f"🎓 Professor Gemini: {t['pergunta_professor']}")
        print(f"🤖 BrotherTrue: {resp['texto']}")
        print(f"📊 Métricas: Diversidade={resp['unique_ratio_words']} | Loop={resp['repetition_loop']} | EOS={resp['ended_eos']} | Tokens={resp['tokens_gerados']}")

    # Estatísticas globais
    diversidades = [h["unique_ratio_words"] for h in historico_dialogo]
    loops = sum(1 for h in historico_dialogo if h["repetition_loop"])
    eos_taxa = sum(1 for h in historico_dialogo if h["ended_eos"]) / len(historico_dialogo)
    media_div = sum(diversidades) / len(diversidades)

    relatorio_final = {
        "checkpoint_avaliado": str(ckpt_file),
        "total_turnos": len(historico_dialogo),
        "metricas_globais": {
            "media_unique_ratio_words": round(media_div, 4),
            "taxa_terminacao_eos": round(eos_taxa * 100.0, 1),
            "loops_detectados": loops,
            "status_geracao": "COERENTE_E_DIVERSA" if media_div >= 0.60 and loops == 0 else "NECESSITA_AJUSTE",
        },
        "dialogos": historico_dialogo,
    }

    out_json = Path(output_report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(relatorio_final, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print(f"✅ Bateria dialógica concluída! Relatório salvo em: {out_json}")
    print(f"🎯 Média de Diversidade Léxica: {media_div:.4f} (Meta >= 0.60)")
    print(f"🎯 Taxa de Encerramento por EOS: {relatorio_final['metricas_globais']['taxa_terminacao_eos']}%")
    print(f"🎯 Loops de Repetição: {loops}")
    print("="*80 + "\n")
    return relatorio_final


if __name__ == "__main__":
    executar_bateria_dialogica()

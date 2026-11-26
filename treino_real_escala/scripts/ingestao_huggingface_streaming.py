"""Ingestão por Streaming de Datasets do HuggingFace em Português.

Utiliza requisições parciais HTTP Range ultra-rápidas para extrair diálogos
em português sem salvar arquivos massivos no SSD local, respeitando os limites
do Sentinela de Hardware.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Importa o sentinela local
sys.path.insert(0, str(Path(__file__).parent))
from hardware_sentry import HardwareSentry


def decode_json_string(s: str) -> str:
    """Decodifica com segurança strings com escapes JSON."""
    try:
        return json.loads(f'"{s}"')
    except Exception:
        # Fallback de decodificação de escapes comuns
        return (
            s.replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\\\", "\\")
        )


def is_quality_conversational(human_text: str, gpt_text: str) -> bool:
    """Valida se o par conversacional possui alta qualidade dialógica e textual."""
    if not human_text or not gpt_text:
        return False

    # Limites para o decodificador de 50M
    if len(human_text) < 20 or len(human_text) > 500:
        return False
    if len(gpt_text) < 40 or len(gpt_text) > 750:
        return False

    # Filtra códigos brutos ou rotinas de programação
    code_indicators = [
        "```", "SELECT ", "import ", "def ", "<html", "function(", "<?php",
        "std::", "public class", "dataframe", "library(", "console.log"
    ]
    if any(ind in human_text for ind in code_indicators):
        return False
    if any(ind in gpt_text for ind in code_indicators):
        return False

    # Verifica acentuação portuguesa típica
    pt_pattern = re.compile(r"[áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ]", re.IGNORECASE)
    if not pt_pattern.search(human_text + gpt_text):
        return False

    return True


def executar_ingestao_huggingface(
    target_count: int = 100,
    output_path: str = "treino_real_escala/dados/hf_dialogos_pt.jsonl",
):
    """Executa a extração streaming em blocos Range de 10 MB."""
    sentry = HardwareSentry()
    status = sentry.throttle()
    print(f"[Sentinela] Status antes da ingestão: SSD Livre = {status['disk_free_gb']} GB")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-portuguese/resolve/main/evol-instruct-portuguese.json"
    print(f"[Ingestão HF] Conectando via Range Request (10 MB) a: {url}")

    headers = {"Range": "bytes=0-10485760"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()

    raw_text = r.text
    print(f"[Ingestão HF] Recebidos {len(r.content) / (1024*1024):.2f} MB. Processando pares dialógicos...")

    # Regex para capturar pares human e gpt
    pattern = re.compile(
        r'\"from\":\s*\"human\",\s*\"value\":\s*\"(.*?)\".*?\"from\":\s*\"gpt\",\s*\"value\":\s*\"(.*?)\"',
        re.DOTALL
    )

    collected = []
    for match in pattern.finditer(raw_text):
        human_raw = match.group(1)
        gpt_raw = match.group(2)

        human_text = decode_json_string(human_raw).strip()
        gpt_text = decode_json_string(gpt_raw).strip()

        if is_quality_conversational(human_text, gpt_text):
            chatml_text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
            vetor_humor = [0.1, 0.05, 0.2, 0.0, 0.1, 0.05, 0.0, 0.1, 0.2, 0.1, 0.0]

            collected.append({
                "id": f"hf_evol_{len(collected):03d}",
                "fonte": "HuggingFace_FreedomIntelligence_evol_instruct_pt",
                "dominio": "geral_conversacional_pt",
                "prompt": human_text,
                "resposta": gpt_text,
                "text": chatml_text,
                "vetor_humor": vetor_humor,
            })

            if len(collected) >= target_count:
                break

    # Se precisar de mais para atingir target_count, faz um segundo range request (10-20MB)
    if len(collected) < target_count:
        print(f"[Ingestão HF] {len(collected)} diálogos coletados no primeiro bloco. Baixando bloco adicional (10-20 MB)...")
        headers = {"Range": "bytes=10485761-20971520"}
        r2 = requests.get(url, headers=headers, timeout=25)
        if r2.status_code == 206:
            raw_text_2 = r2.text
            for match in pattern.finditer(raw_text_2):
                human_text = decode_json_string(match.group(1)).strip()
                gpt_text = decode_json_string(match.group(2)).strip()
                if is_quality_conversational(human_text, gpt_text):
                    chatml_text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
                    vetor_humor = [0.1, 0.05, 0.2, 0.0, 0.1, 0.05, 0.0, 0.1, 0.2, 0.1, 0.0]
                    collected.append({
                        "id": f"hf_evol_{len(collected):03d}",
                        "fonte": "HuggingFace_FreedomIntelligence_evol_instruct_pt",
                        "dominio": "geral_conversacional_pt",
                        "prompt": human_text,
                        "resposta": gpt_text,
                        "text": chatml_text,
                        "vetor_humor": vetor_humor,
                    })
                    if len(collected) >= target_count:
                        break

    # Salva dataset
    with open(out_file, "w", encoding="utf-8") as f:
        for item in collected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    sentry.purge_caches()
    final_status = sentry.throttle()
    print(f"[Ingestão HF] Sucesso! {len(collected)} diálogos conversacionais salvos em {out_file} ({out_file.stat().st_size / 1024:.1f} KB).")
    print(f"[Sentinela] SSD Livre final: {final_status['disk_free_gb']} GB")
    return len(collected)


if __name__ == "__main__":
    count = executar_ingestao_huggingface(target_count=100)
    print(f"[OK] Ingestão HF concluída com {count} registros.")

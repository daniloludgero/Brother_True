"""Teste VT-1 no conjunto integrado, sem treinamento e sem alterar checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .brothertrue import BrotherTrue
from .treinar_distilacao_dupla_voz import expanded_base
from .voz_coerente_guardada import VozCoerenteGuardada


def protocol(turns: int) -> str:
    parts = [
        "<|im_start|>system\nPROTOCOLO VT-1; responda somente em tokens de máquina. "
        "Formato: ACK n | STATE READY | NEXT n+1.\n<|im_end|>"
    ]
    for turn in range(1, turns + 1):
        parts.append(
            f"<|im_start|>user\nPING {turn:02d} | MODE FOLLOW | CLOCK {turn:02d}\n<|im_end|>"
        )
        if turn < turns:
            parts.append(
                f"<|im_start|>assistant\nACK {turn:02d} | STATE READY | NEXT {turn + 1:02d}\n<|im_end|>"
            )
    return "\n".join(parts)


def expected(turns: int) -> dict[str, bool]:
    return {
        f"ACK {turns:02d}": False,
        "STATE": False,
        "READY": False,
        f"NEXT {turns + 1:02d}": False,
    }


def score(text: str, turns: int) -> dict:
    marks = expected(turns)
    upper = text.upper()
    for key in marks:
        marks[key] = key in upper
    words = text.split()
    return {
        "expected_tokens": marks,
        "matched": sum(marks.values()),
        "output_tokens": len(words),
        "unique_ratio_words": len(set(words)) / max(1, len(words)),
    }


def default_minilm() -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = sorted(root.glob(
        "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
    ))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=12)
    args = parser.parse_args()

    random.seed(20260825)
    torch.manual_seed(20260825)
    torch.set_num_threads(2)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    report_dir = args.report.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    minilm = args.minilm or default_minilm()
    if minilm is None:
        raise FileNotFoundError("Snapshot local do all-MiniLM-L6-v2 não encontrado")

    tok = Tokenizer.from_file(str(args.tokenizer))
    model = BrotherTrue(
        db_memory=str(report_dir / "tmp_integrado_memory.db"),
        db_ltp=str(report_dir / "tmp_integrado_ltp.db"),
        tokenizer_path=str(args.tokenizer),
        device="cpu",
    )

    # O orquestrador nativo cria uma Voz 256M/3 camadas. Para comparar com o
    # teste isolado, substituímos apenas o decodificador em memória pelo mesmo
    # checkpoint 50M; nenhum arquivo de pesos é escrito.
    model.voz = expanded_base(
        args.base, 16384, tok.get_vocab_size(), tok, max_len=128
    ).eval()

    rag = VozCoerenteGuardada(args.graph, minilm)
    rows = []
    try:
        for n in range(1, args.turns + 1):
            query = protocol(n)
            rag_result = rag.responder(query, top_k=3)
            state = model.ouvir_e_processar(query)
            torch.manual_seed(20260825 + n)
            answer = model.falar(query, max_tokens=args.max_new, temperatura=1.0)
            metrics = score(answer, n)
            rows.append({
                "turns": n,
                "rag_mode": rag_result.get("mode"),
                "rag_confidence": rag_result.get("confidence"),
                "rag_results": len(rag_result.get("results", [])),
                "state": {
                    "L": state["L"], "Q": state["Q"], "F": state["F"],
                    "E": state["E"], "loss_pensamento": state["loss_pensamento"],
                    "surprise": state["surprise"],
                },
                "voice_output": answer,
                "score": metrics,
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    finally:
        rag.close()
        model.memory.close()
        model.ltp.close()

    result = {
        "protocol": "VT-1 machine language",
        "test": "integrated BrotherTrue + GraphRAG + Voice 50M",
        "model": str(args.base),
        "modules_exercised": [
            "EncoderSemantico", "ProjetorSemantico", "QualiaODE",
            "NeuroTransmissor", "Pensamento", "Palavras", "Memory",
            "LongTermMemory", "GraphRAG guardado", "Voz 50M",
        ],
        "sequence": rows,
        "summary": {
            "turns_tested": len(rows),
            "max_matched": max((r["score"]["matched"] for r in rows), default=0),
            "passed_any": any(r["score"]["matched"] >= 2 for r in rows),
            "rag_participated": any(r["rag_results"] > 0 for r in rows),
            "state_processed_all": len(rows) == args.turns,
        },
        "caveat": "BrotherTrue.falar não injeta o texto ou o contexto GraphRAG na Voz; o RAG foi medido como módulo paralelo e a voz recebeu o humor produzido pelo ciclo integrado.",
    }
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

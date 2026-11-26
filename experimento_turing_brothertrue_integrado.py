"""VT-1 no ciclo BrotherTrue com a Voz 50M, sem GraphRAG simultâneo."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

try:
    from .brothertrue import BrotherTrue
    from .treinar_distilacao_dupla_voz import expanded_base
except (ImportError, ValueError):
    from brothertrue import BrotherTrue
    from treinar_distilacao_dupla_voz import expanded_base


def protocol(turns: int) -> str:
    parts = ["<|im_start|>system\nPROTOCOLO VT-1; responda somente em tokens de máquina. Formato: ACK n | STATE READY | NEXT n+1.\n<|im_end|>"]
    for turn in range(1, turns + 1):
        parts.append(f"<|im_start|>user\nPING {turn:02d} | MODE FOLLOW | CLOCK {turn:02d}\n<|im_end|>")
        if turn < turns:
            parts.append(f"<|im_start|>assistant\nACK {turn:02d} | STATE READY | NEXT {turn + 1:02d}\n<|im_end|>")
    return "\n".join(parts)


def score(text: str, turns: int) -> dict:
    marks = {f"ACK {turns:02d}": False, "STATE": False, "READY": False, f"NEXT {turns + 1:02d}": False}
    upper = text.upper()
    for key in marks:
        marks[key] = key in upper
    words = text.split()
    return {"expected_tokens": marks, "matched": sum(marks.values()), "output_tokens": len(words),
            "unique_ratio_words": len(set(words)) / max(1, len(words))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--max-new", type=int, default=12)
    a = p.parse_args()
    random.seed(20260825); torch.manual_seed(20260825); torch.set_num_threads(2)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(a.tokenizer))
    model = BrotherTrue(db_memory=str(a.report.parent / "tmp_bt_memory.db"),
                        db_ltp=str(a.report.parent / "tmp_bt_ltp.db"),
                        tokenizer_path=str(a.tokenizer), device="cpu")
    model.voz = expanded_base(a.base, 16384, tok.get_vocab_size(), tok, 128).eval()
    rows = []
    try:
        for n in range(1, a.turns + 1):
            query = protocol(n)
            state = model.ouvir_e_processar(query)
            torch.manual_seed(20260825 + n)
            answer = model.falar(query, max_tokens=a.max_new, temperatura=1.0)
            row = {"turns": n,
                   "state": {k: state[k] for k in ("L", "Q", "F", "E", "loss_pensamento", "surprise")},
                   "voice_output": answer, "score": score(answer, n)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        model.memory.close(); model.ltp.close()
    result = {"protocol": "VT-1 machine language", "test": "integrated BrotherTrue state-to-voice",
              "model": str(a.base),
              "modules_exercised": ["EncoderSemantico", "ProjetorSemantico", "QualiaODE", "NeuroTransmissor",
                                    "Pensamento", "Palavras", "Memory", "LongTermMemory", "Voz 50M"],
              "sequence": rows,
              "summary": {"turns_tested": len(rows), "max_matched": max((r["score"]["matched"] for r in rows), default=0),
                          "passed_any": any(r["score"]["matched"] >= 2 for r in rows),
                          "state_processed_all": len(rows) == a.turns},
              "caveat": "GraphRAG não participa do método BrotherTrue.falar atual; foi avaliado separadamente.",}
    a.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

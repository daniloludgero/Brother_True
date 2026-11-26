"""VT-1 integrado; adapta localmente o tensor de inferência do MiniLM."""
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
    from .experimento_turing_brothertrue_integrado import protocol, score
    from .treinar_distilacao_dupla_voz import expanded_base
except (ImportError, ValueError):
    from brothertrue import BrotherTrue
    from experimento_turing_brothertrue_integrado import protocol, score
    from treinar_distilacao_dupla_voz import expanded_base


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
    # Correção de fronteira exclusiva do teste: o encoder retorna tensor de
    # inference_mode e o projetor do projeto ainda usa autograd.
    if model.encoder_sem is not None:
        original_encode = model.encoder_sem.codificar
        model.encoder_sem.codificar = lambda text: original_encode(text).clone()
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
              "model": str(a.base), "in_test_adapter": "clone MiniLM inference tensor",
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

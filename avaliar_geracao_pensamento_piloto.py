"""Comparação generativa curta entre o checkpoint-base e o piloto."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pensamento_linguistico import PensamentoLinguistico


def load_model(path: Path, vocab: int, max_len: int):
    model = PensamentoLinguistico(vocab_size=vocab, d_model=384, nhead=6,
                                  num_layers=4, dim_feedforward=1536,
                                  max_len=max_len)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model", payload))
    model.load_state_dict(state, strict=False)
    return model.eval()


def metrics(tok, ids, eos):
    seq = ids[0].tolist()
    text = tok.decode(seq, skip_special_tokens=True).strip()
    adjacent = sum(a == b for a, b in zip(seq, seq[1:]))
    return {
        "token_count": len(seq),
        "text_chars": len(text),
        "nonempty": bool(text),
        "eos_present": eos in seq,
        "adjacent_repeat_ratio": adjacent / max(1, len(seq) - 1),
        "unique_token_ratio": len(set(seq)) / max(1, len(seq)),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    tok = Tokenizer.from_file(str(args.tokenizer))
    humor = torch.zeros((1, 11), dtype=torch.float32)
    models = {"base": load_model(args.base, tok.get_vocab_size(), 128),
              "pilot": load_model(args.pilot, tok.get_vocab_size(), 128)}
    report = {"experiment": "pensamento_generation_smoke_test", "samples": args.samples,
              "max_new": args.max_new, "models": {}}
    for name, model in models.items():
        samples = []
        for i in range(args.samples):
            torch.manual_seed(20260827 + i)
            ids = model.gerar(humor, max_novos_tokens=args.max_new,
                              temperatura=0.8, top_k=20, top_p=0.9)
            samples.append(metrics(tok, ids, model.eos_token_id))
        report["models"][name] = {
            "nonempty_rate": sum(x["nonempty"] for x in samples) / len(samples),
            "eos_rate": sum(x["eos_present"] for x in samples) / len(samples),
            "mean_tokens": sum(x["token_count"] for x in samples) / len(samples),
            "mean_adjacent_repeat_ratio": sum(x["adjacent_repeat_ratio"] for x in samples) / len(samples),
            "mean_unique_token_ratio": sum(x["unique_token_ratio"] for x in samples) / len(samples),
            "samples": samples,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

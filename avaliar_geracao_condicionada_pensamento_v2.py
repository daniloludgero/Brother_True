"""Versão corrigida: mede a continuação após o prompt efetivamente truncado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pensamento_linguistico import PensamentoLinguistico


def load_model(path, vocab):
    m = PensamentoLinguistico(vocab_size=vocab, d_model=384, nhead=6,
                              num_layers=4, dim_feedforward=1536, max_len=128)
    p = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(p.get("model_state_dict", p.get("model", p)), strict=False)
    return m.eval()


def generate(model, tok, prompt, max_new):
    raw = list(tok.encode(prompt + "\n<think>\n").ids)
    ids = ([model.bos_token_id] + raw)[-(model.max_len - 1):]
    prefix_len = len(ids)
    humor = torch.zeros((1, 11), dtype=torch.float32)
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.long), humor)[0, -1] / 0.8
            nxt = int(torch.argmax(logits).item())
        ids.append(nxt)
        if nxt == model.eos_token_id or len(ids) >= model.max_len:
            break
    continuation = ids[prefix_len:]
    return continuation, tok.decode(continuation, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()
    rows = []
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("prompt") and row.get("think"):
                    rows.append(row)
                if len(rows) >= args.samples:
                    break
    tok = Tokenizer.from_file(str(args.tokenizer))
    models = {"base": load_model(args.base, tok.get_vocab_size()),
              "pilot": load_model(args.pilot, tok.get_vocab_size())}
    report = {"experiment": "pensamento_prompt_conditioned_smoke_test_v2",
              "samples": len(rows), "max_new": args.max_new, "models": {}}
    for name, model in models.items():
        results = []
        for row in rows:
            out_ids, text = generate(model, tok, row["prompt"], args.max_new)
            target = set(tok.encode(row["think"]).ids)
            results.append({"tokens": len(out_ids), "nonempty": bool(text),
                            "eos": model.eos_token_id in out_ids,
                            "target_token_overlap": len(set(out_ids) & target) / max(1, len(set(out_ids)))})
        report["models"][name] = {
            "nonempty_rate": sum(r["nonempty"] for r in results) / len(results),
            "eos_rate": sum(r["eos"] for r in results) / len(results),
            "mean_tokens": sum(r["tokens"] for r in results) / len(results),
            "mean_target_token_overlap": sum(r["target_token_overlap"] for r in results) / len(results),
            "samples": results,
        }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

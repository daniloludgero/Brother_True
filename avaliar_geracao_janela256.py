"""Comparação generativa do checkpoint-base e do piloto com janela 256."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pensamento_linguistico import PensamentoLinguistico


def load_model(path, vocab):
    m = PensamentoLinguistico(vocab_size=vocab, d_model=384, nhead=6,
                              num_layers=4, dim_feedforward=1536, max_len=256)
    p = torch.load(path, map_location="cpu", weights_only=False)
    state = p.get("model_state_dict", p.get("model", p))
    old = state.pop("pos_embedding.weight", None)
    m.load_state_dict(state, strict=False)
    if old is not None:
        with torch.no_grad():
            m.pos_embedding.weight.copy_(F.interpolate(old.T.unsqueeze(0), size=256,
                                                        mode="linear", align_corners=False).squeeze(0).T)
    return m.eval()


def generate(model, tok, prompt, max_new):
    raw = list(tok.encode(prompt + "\n<think>\n").ids)
    ids = [model.bos_token_id] + raw[: model.max_len - max_new - 1]
    prefix = len(ids); humor = torch.zeros((1, 11), dtype=torch.float32)
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.long), humor)[0, -1]
            nxt = int(torch.argmax(logits).item())
        ids.append(nxt)
        if nxt == model.eos_token_id:
            break
    return ids[prefix:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True); ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=24); args = ap.parse_args()
    rows = []
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("prompt") and r.get("think"): rows.append(r)
                if len(rows) >= args.samples: break
    tok = Tokenizer.from_file(str(args.tokenizer))
    models = {"base": load_model(args.base, tok.get_vocab_size()),
              "pilot": load_model(args.pilot, tok.get_vocab_size())}
    report = {"experiment": "pensamento_window256_generation", "samples": len(rows),
              "max_new": args.max_new, "models": {}}
    for name, model in models.items():
        results = []
        for row in rows:
            ids = generate(model, tok, row["prompt"], args.max_new)
            text = tok.decode(ids, skip_special_tokens=True).strip()
            target = set(tok.encode(row["think"]).ids)
            results.append({"tokens": len(ids), "nonempty": bool(text),
                            "eos": model.eos_token_id in ids,
                            "target_token_overlap": len(set(ids) & target) / max(1, len(set(ids))),
                            "adjacent_repeat_ratio": sum(a == b for a, b in zip(ids, ids[1:])) / max(1, len(ids) - 1)})
        report["models"][name] = {
            "nonempty_rate": sum(r["nonempty"] for r in results) / len(results),
            "eos_rate": sum(r["eos"] for r in results) / len(results),
            "mean_tokens": sum(r["tokens"] for r in results) / len(results),
            "mean_target_token_overlap": sum(r["target_token_overlap"] for r in results) / len(results),
            "mean_adjacent_repeat_ratio": sum(r["adjacent_repeat_ratio"] for r in results) / len(results),
            "samples": results,
        }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

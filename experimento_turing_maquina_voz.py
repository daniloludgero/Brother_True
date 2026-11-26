"""Teste isolado de acompanhamento temporal em protocolo de máquina."""

from __future__ import annotations

import argparse, json, re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .checkpoint_manager_safe import load_checkpoint
from .treinar_distilacao_dupla_voz import expanded_base
from . import treinar_lora_voz

treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)

TURN_TEMPLATE = (
    "<|im_start|>system\nPROTOCOLO VT-1; responda somente em tokens de máquina. "
    "Formato: ACK n | STATE READY | NEXT n+1.<|im_end|>\n"
)


def protocol_prefix(turns: int) -> str:
    text = TURN_TEMPLATE
    for i in range(1, turns + 1):
        text += f"<|im_start|>user\nPING {i:02d} | MODE FOLLOW | CLOCK {i:02d}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        if i < turns:
            text += f"ACK {i:02d} | STATE READY | NEXT {i+1:02d}<|im_end|>\n"
    return text


@torch.no_grad()
def generate(model, tok, prefix: str, max_len: int, max_new: int):
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    im_end = tok.token_to_id("<|im_end|>")
    end_text = tok.token_to_id("<|end_of_text|>")
    ids = tok.encode(prefix, add_special_tokens=False).ids[-(max_len - max_new - 1):]
    x = torch.tensor([[bos] + ids], dtype=torch.long)
    generated = []
    for _ in range(max_new):
        nxt = int(model(x, torch.zeros(1, 11))[0, -1].argmax())
        generated.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        if nxt in (eos, im_end, end_text) or x.size(1) >= max_len:
            break
    text = tok.decode(generated, skip_special_tokens=True)
    return generated, text


def score(text: str, turns: int):
    expected = [f"ACK {turns:02d}", "STATE", "READY", f"NEXT {turns+1:02d}"]
    present = {item: item.lower() in text.lower() for item in expected}
    tokens = re.findall(r"\S+", text.lower())
    unique = len(set(tokens)) / max(1, len(tokens))
    return {"expected_tokens": present, "matched": sum(present.values()),
            "unique_ratio_words": unique, "output_tokens": len(tokens)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-new", type=int, default=24)
    a = p.parse_args()
    torch.set_num_threads(2)
    tok = Tokenizer.from_file(str(a.tokenizer))
    model = expanded_base(a.base, 16384, tok.get_vocab_size(), tok, a.max_len)
    load_checkpoint(a.base, model=model, map_location="cpu")
    model.eval()
    rows = []
    for turns in (1, 2, 3, 4, 5, 6):
        prefix = protocol_prefix(turns)
        ids, text = generate(model, tok, prefix, a.max_len, a.max_new)
        rows.append({"turns": turns, "prefix_chars": len(prefix),
                     "completion": text, "score": score(text, turns),
                     "ended_special": bool(ids and ids[-1] in {
                         tok.token_to_id("<eos>"), tok.token_to_id("<|im_end|>"),
                         tok.token_to_id("<|end_of_text|>")})})
    report = {"protocol": "VT-1 machine language", "model": str(a.base),
              "interpretation": "upper-bound prefix-following test; public Voz.gerar does not accept text context",
              "sequence": rows,
              "summary": {"max_matched": max(x["score"]["matched"] for x in rows),
                          "turns_tested": len(rows),
                          "passed_any": any(x["score"]["matched"] >= 2 for x in rows)}}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()

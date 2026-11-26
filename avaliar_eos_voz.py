"""Compara EOS-aware com SFT anterior, com e sem guarda de comprimento mínimo."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from . import avaliar_sft_voz
from .checkpoint_manager_safe import load_checkpoint
from .treino_voz_texto_limpo import offsets


@torch.no_grad()
def generate_guarded(model, tok, prompt, max_len=128, max_new=48, min_new=8):
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    prompt_ids = prompt_ids[-(max_len - max_new - 1):]
    ids = [bos] + prompt_ids
    generated = []
    input_ids = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        logits = model(input_ids, torch.zeros(1, 11))[0, -1].clone()
        if len(generated) < min_new:
            logits[eos] = -float("inf")
        next_id = int(logits.argmax())
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        if next_id == eos or input_ids.size(1) >= max_len:
            break
    return {"completion": tok.decode(generated, skip_special_tokens=True),
            "new_tokens": len(generated),
            "ended_eos": bool(generated and generated[-1] == eos),
            "unique_ratio": len(set(generated))/max(1, len(generated))}


def evaluate_adapter(label, adapter, args, positions, tok):
    model = avaliar_sft_voz.load_adapter(tok, args.base, adapter, args.max_len)
    loss = avaliar_sft_voz.response_loss(model, args.data, positions, tok, args.max_len)
    examples = []
    for position in positions[:4]:
        with args.data.open("rb") as handle:
            handle.seek(position)
            item = json.loads(handle.readline().decode("utf-8"))
        examples.append({"reference": item["response"],
                         "normal": avaliar_sft_voz.generate(model, tok, item["prompt"], args.max_len, 48),
                         "guarded": generate_guarded(model, tok, item["prompt"], args.max_len, 48, 8)})
    return {"label": label, "response_loss": loss, "examples": examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--old-adapter", type=Path, required=True)
    p.add_argument("--eos-adapter", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = avaliar_sft_voz.Tokenizer.from_file(str(args.tokenizer))
    all_positions = offsets(args.data)
    order = list(range(len(all_positions))); random.Random(20260824).shuffle(order)
    positions = [all_positions[i] for i in order[:8]]
    result = {"old_sft": evaluate_adapter("sft_step525", args.old_adapter, args, positions, tok),
              "eos_aware": evaluate_adapter("eos_aware_step800", args.eos_adapter, args, positions, tok)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

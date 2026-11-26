"""Correção do avaliador SFT: converte índices de validação em offsets reais."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from . import avaliar_sft_voz
from .checkpoint_manager_safe import load_checkpoint
from .treino_voz_texto_limpo import offsets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--samples", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = avaliar_sft_voz.Tokenizer.from_file(str(a.tokenizer))
    all_positions = offsets(a.data)
    order = list(range(len(all_positions)))
    random.Random(20260824).shuffle(order)
    val_positions = [all_positions[i] for i in order[:max(1, len(order) // 10)][:a.samples]]

    base_model = avaliar_sft_voz.make_model(tok, a.max_len)
    load_checkpoint(a.base, model=base_model, map_location="cpu")
    base_loss = avaliar_sft_voz.response_loss(base_model, a.data, val_positions, tok, a.max_len)
    del base_model

    model = avaliar_sft_voz.load_adapter(tok, a.base, a.adapter, a.max_len)
    lora_loss = avaliar_sft_voz.response_loss(model, a.data, val_positions, tok, a.max_len)
    examples = []
    for position in val_positions:
        with a.data.open("rb") as handle:
            handle.seek(position)
            item = json.loads(handle.readline().decode("utf-8"))
        examples.append({"prompt": item["prompt"], "reference": item["response"],
                         "generated": avaliar_sft_voz.generate(
                             model, tok, item["prompt"], a.max_len, 48)})
    result = {"model": "Voz 50M SFT LoRA", "base_response_loss": base_loss,
              "sft_response_loss": lora_loss, "examples": examples}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Valida o carregamento do adaptador LoRA separado sobre a base congelada."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from . import treinar_lora_voz
from .checkpoint_manager_safe import load_checkpoint
from .treino_voz_texto_limpo import offsets, sample
from .voz import Voz


treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


def make_model(tok: Tokenizer, max_len: int) -> Voz:
    return Voz(vocab_size=tok.get_vocab_size(), d_model=512, nhead=8,
               num_layers=8, dim_feedforward=2048, max_len=max_len,
               pad_token_id=tok.token_to_id("<pad>") or 0,
               bos_token_id=tok.token_to_id("<bos>") or 1,
               eos_token_id=tok.token_to_id("<eos>") or 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=128)
    a = p.parse_args()

    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(a.tokenizer))
    base = make_model(tok, a.max_len)
    load_checkpoint(a.base, model=base, map_location="cpu")
    treinar_lora_voz.add_lora(base, rank=8, alpha=16.0, dropout=0.05)
    load_checkpoint(a.full, model=base, map_location="cpu")
    base.eval()

    positions = offsets(a.data)
    item = sample(a.data, positions[0], tok, a.max_len,
                  tok.token_to_id("<bos>") or 1, tok.token_to_id("<eos>") or 2)
    if item is None:
        raise RuntimeError("A primeira amostra não produziu tokens")
    x, _ = item
    with torch.no_grad():
        full_logits = base(x.unsqueeze(0), torch.zeros(1, 11))
    del base

    adapter_model = make_model(tok, a.max_len)
    load_checkpoint(a.base, model=adapter_model, map_location="cpu")
    treinar_lora_voz.add_lora(adapter_model, rank=8, alpha=16.0, dropout=0.05)
    adapter_payload = torch.load(a.adapter, map_location="cpu", weights_only=False)
    missing, unexpected = adapter_model.load_state_dict(
        adapter_payload["adapter_state_dict"], strict=False
    )
    adapter_model.eval()
    with torch.no_grad():
        adapter_logits = adapter_model(x.unsqueeze(0), torch.zeros(1, 11))
    difference = float((full_logits - adapter_logits).abs().max())
    result = {
        "adapter_format": adapter_payload.get("format"),
        "adapter_step": adapter_payload.get("step"),
        "trainable_parameters": adapter_payload.get("trainable_parameters"),
        "missing_base_keys_expected": len(missing),
        "unexpected_keys": len(unexpected),
        "max_logit_difference": difference,
        "equivalent": difference < 1e-5 and not unexpected,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["equivalent"]:
        raise SystemExit("Adaptador separado não reproduziu o checkpoint completo")


if __name__ == "__main__":
    main()

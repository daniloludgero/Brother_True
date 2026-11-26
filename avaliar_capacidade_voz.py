"""Avalia capacidade linguística direta da Voz 50M e do adaptador LoRA."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
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


def build_eval_items(data: Path, tok: Tokenizer, max_len: int, count: int):
    positions = offsets(data)
    order = list(range(len(positions)))
    random.Random(20260824).shuffle(order)
    val = order[:max(1, len(order) // 10)]
    random.seed(20260824 + 99)
    items = []
    for index in val[:count]:
        item = sample(data, positions[index], tok, max_len,
                      tok.token_to_id("<bos>") or 1,
                      tok.token_to_id("<eos>") or 2)
        if item is not None:
            items.append(item)
    return items


@torch.no_grad()
def loss_metrics(model: Voz, items):
    model.eval()
    losses = []
    for x, y in items:
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y)))
    value = sum(losses) / max(1, len(losses))
    return {"loss": value, "perplexity": math.exp(min(value, 20.0)),
            "samples": len(losses)}


@torch.no_grad()
def greedy_completion(model: Voz, tok: Tokenizer, text: str, max_new: int,
                      max_len: int):
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    ids = [bos] + tok.encode(text, add_special_tokens=False).ids
    ids = ids[-(max_len - max_new):]
    input_ids = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        logits = model(input_ids, torch.zeros(1, 11))
        next_id = int(logits[0, -1].argmax())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        if next_id == eos or input_ids.size(1) >= max_len:
            break
    decoded = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
    generated = input_ids[0].tolist()[len(ids):]
    unique = len(set(generated)) / max(1, len(generated))
    repeats = 1.0 - unique
    return {"text": decoded, "new_tokens": len(generated),
            "unique_token_ratio": unique, "repetition_ratio": repeats,
            "ended_eos": bool(generated and generated[-1] == eos)}


def load_base(tok: Tokenizer, base: Path, max_len: int) -> Voz:
    model = make_model(tok, max_len)
    load_checkpoint(base, model=model, map_location="cpu")
    return model


def load_adapter(tok: Tokenizer, base: Path, adapter: Path, max_len: int) -> Voz:
    model = load_base(tok, base, max_len)
    treinar_lora_voz.add_lora(model, rank=8, alpha=16.0, dropout=0.05)
    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["adapter_state_dict"], strict=False)
    if unexpected or not any("lora_a" in key for key in payload["adapter_state_dict"]):
        raise RuntimeError(f"Adaptador inválido: unexpected={unexpected}")
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--samples", type=int, default=16)
    args = p.parse_args()

    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(args.tokenizer))
    items = build_eval_items(args.data, tok, args.max_len, args.samples)
    prompts = [
        "Olá, eu sou",
        "Uma conversa clara deve",
        "The purpose of a good conversation is",
        "If a function receives a number",
    ]
    result = {"model": "Voz 50M", "evaluation_type": "direct_causal_lm",
              "note": "pré-treinamento causal; não é benchmark de instruções",
              "base": {}, "lora": {}}

    base_model = load_base(tok, args.base, args.max_len)
    result["base"]["parameters"] = sum(p.numel() for p in base_model.parameters())
    result["base"]["loss"] = loss_metrics(base_model, items)
    result["base"]["completions"] = {
        prompt: greedy_completion(base_model, tok, prompt, 32, args.max_len)
        for prompt in prompts
    }
    del base_model

    lora_model = load_adapter(tok, args.base, args.adapter, args.max_len)
    result["lora"]["parameters"] = sum(p.numel() for p in lora_model.parameters())
    result["lora"]["trainable_parameters"] = sum(
        p.numel() for p in lora_model.parameters() if p.requires_grad
    )
    result["lora"]["loss"] = loss_metrics(lora_model, items)
    result["lora"]["completions"] = {
        prompt: greedy_completion(lora_model, tok, prompt, 32, args.max_len)
        for prompt in prompts
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

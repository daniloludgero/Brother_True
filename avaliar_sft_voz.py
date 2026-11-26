"""Avalia respostas no mesmo formato conversacional usado pelo SFT."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from . import treinar_lora_voz, treinar_sft_lora_voz
from .checkpoint_manager_safe import load_checkpoint
from .treino_voz_texto_limpo import offsets
from .voz import Voz


treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


def make_model(tok, max_len):
    return Voz(vocab_size=tok.get_vocab_size(), d_model=512, nhead=8,
               num_layers=8, dim_feedforward=2048, max_len=max_len,
               pad_token_id=tok.token_to_id("<pad>") or 0,
               bos_token_id=tok.token_to_id("<bos>") or 1,
               eos_token_id=tok.token_to_id("<eos>") or 2)


def load_adapter(tok, base, adapter, max_len):
    model = make_model(tok, max_len)
    load_checkpoint(base, model=model, map_location="cpu")
    treinar_lora_voz.add_lora(model, rank=8, alpha=16.0, dropout=0.05)
    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["adapter_state_dict"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def response_loss(model, data, positions, tok, max_len):
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    values = []
    for position in positions:
        item = treinar_sft_lora_voz.read_example(data, position, tok, max_len, bos, eos)
        if item is None:
            continue
        x, labels = item
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        values.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            labels, ignore_index=-100)))
    loss = sum(values) / max(1, len(values))
    return {"loss": loss, "perplexity": math.exp(min(loss, 20.0)),
            "samples": len(values)}


@torch.no_grad()
def generate(model, tok, prompt, max_len, max_new):
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    prompt_ids = prompt_ids[-(max_len - max_new - 1):]
    ids = [bos] + prompt_ids
    generated = []
    input_ids = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        logits = model(input_ids, torch.zeros(1, 11))
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        if next_id == eos or input_ids.size(1) >= max_len:
            break
    return {"completion": tok.decode(generated, skip_special_tokens=True),
            "new_tokens": len(generated),
            "unique_ratio": len(set(generated)) / max(1, len(generated)),
            "ended_eos": bool(generated and generated[-1] == eos)}


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
    tok = Tokenizer.from_file(str(a.tokenizer))
    positions = offsets(a.data)
    order = list(range(len(positions)))
    random.Random(20260824).shuffle(order)
    val_positions = order[:max(1, len(order) // 10)][:a.samples]

    base_model = make_model(tok, a.max_len)
    load_checkpoint(a.base, model=base_model, map_location="cpu")
    base_loss = response_loss(base_model, a.data, val_positions, tok, a.max_len)
    del base_model

    model = load_adapter(tok, a.base, a.adapter, a.max_len)
    lora_loss = response_loss(model, a.data, val_positions, tok, a.max_len)
    examples = []
    for position in val_positions:
        with a.data.open("rb") as handle:
            handle.seek(position)
            item = json.loads(handle.readline().decode("utf-8"))
        generated = generate(model, tok, item["prompt"], a.max_len, 48)
        examples.append({"prompt": item["prompt"], "reference": item["response"],
                         "generated": generated})
    result = {"model": "Voz 50M SFT LoRA", "base_response_loss": base_loss,
              "sft_response_loss": lora_loss, "examples": examples}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

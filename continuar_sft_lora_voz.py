"""Continua SFT LoRA com loss somente na resposta do assistente."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from . import treinar_lora_voz, treinar_sft_lora_voz
from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .treino_voz_texto_limpo import offsets
from .voz import Voz


treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--save-every", type=int, default=25)
    a = p.parse_args()
    random.seed(20260824); torch.manual_seed(20260824)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(a.tokenizer))
    pad = tok.token_to_id("<pad>") or 0; bos = tok.token_to_id("<bos>") or 1; eos = tok.token_to_id("<eos>") or 2
    positions = offsets(a.data); order = list(range(len(positions))); random.shuffle(order)
    val = order[:max(1, len(order)//10)]; train = order[len(val):]
    model = Voz(vocab_size=tok.get_vocab_size(), d_model=512, nhead=8, num_layers=8,
                dim_feedforward=2048, max_len=a.max_len, pad_token_id=pad,
                bos_token_id=bos, eos_token_id=eos)
    load_checkpoint(a.base, model=model, map_location="cpu")
    total_params = sum(x.numel() for x in model.parameters())
    trainable = treinar_lora_voz.add_lora(model, a.rank, a.alpha, 0.05)
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    latest, best = a.out/"latest.pt", a.out/"best.pt"
    source = a.resume
    state = load_checkpoint(source, model=model, optimizer=optimizer, map_location="cpu")
    step = int(state.get("step", 0)); best_val = float(state.get("metrics",{}).get("val_loss", float("inf")))
    for group in optimizer.param_groups: group["lr"] = a.lr
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90))
    rng = random.Random(20260824 + step)
    print(json.dumps({"event":"sft_resumed","step":step,"source":str(source),
                      "total_parameters":total_params,"trainable_parameters":trainable}, ensure_ascii=False), flush=True)
    while step < a.steps:
        item = treinar_sft_lora_voz.read_example(a.data, positions[train[rng.randrange(len(train))]], tok, a.max_len, bos, eos)
        if item is None: continue
        x, labels = item; treinar_lora_voz.set_lora_train_mode(model); optimizer.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0), torch.zeros(1,11))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels, ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 1.0); optimizer.step(); step += 1
        if step % a.save_every == 0 or step == a.steps:
            val_loss = treinar_sft_lora_voz.evaluate(model, a.data, [positions[i] for i in val], tok, a.max_len, bos, eos)
            metrics={"train_loss":float(loss.detach()),"val_loss":val_loss,"dataset":"human_chats_sft_response_only",
                     "total_parameters":total_params,"trainable_parameters":trainable,"loss_target":"assistant_response_only","base":str(a.base)}
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step, metrics=metrics)
            if val_loss < best_val:
                best_val=val_loss; save_checkpoint(best, model=model, optimizer=optimizer, step=step, metrics=metrics)
            print(json.dumps({"step":step,**metrics,"best_val":best_val,**governor.throttle()},ensure_ascii=False),flush=True)
    print(f"[OK] SFT continuado até o passo {step}; melhor validação {best_val:.5f}",flush=True)


if __name__ == "__main__": main()

"""Treina LoRA com loss mascarada somente na resposta do assistente."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from . import treinar_lora_voz
from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .treino_voz_texto_limpo import offsets
from .voz import Voz


treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


def read_example(path: Path, position: int, tok: Tokenizer, max_len: int,
                 bos: int, eos: int):
    with path.open("rb") as handle:
        handle.seek(position)
        item = json.loads(handle.readline().decode("utf-8"))
    prompt_ids = tok.encode(item["prompt"], add_special_tokens=False).ids
    response_ids = tok.encode(item["response"], add_special_tokens=False).ids + [eos]
    if not response_ids:
        return None
    response_ids = response_ids[: max_len - 2]
    prompt_budget = max(1, max_len - 1 - len(response_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    sequence = [bos] + prompt_ids + response_ids
    x = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor(sequence[1:], dtype=torch.long)
    response_start = 1 + len(prompt_ids)
    labels[: max(0, response_start - 1)] = -100
    return x, labels


@torch.no_grad()
def evaluate(model, data: Path, positions, tok, max_len, bos, eos):
    model.eval()
    values = []
    for position in positions[:32]:
        item = read_example(data, position, tok, max_len, bos, eos)
        if item is None:
            continue
        x, labels = item
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        values.append(float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels,
            ignore_index=-100,
        )))
    return sum(values) / max(1, len(values))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--save-every", type=int, default=25)
    a = p.parse_args()

    random.seed(20260824)
    torch.manual_seed(20260824)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(a.tokenizer))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    positions = offsets(a.data)
    order = list(range(len(positions)))
    random.shuffle(order)
    val = order[:max(1, len(order) // 10)]
    train = order[len(val):]

    model = Voz(vocab_size=tok.get_vocab_size(), d_model=512, nhead=8,
                num_layers=8, dim_feedforward=2048, max_len=a.max_len,
                pad_token_id=pad, bos_token_id=bos, eos_token_id=eos)
    load_checkpoint(a.base, model=model, map_location="cpu")
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = treinar_lora_voz.add_lora(model, a.rank, a.alpha, 0.05)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=a.lr,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    latest, best = a.out / "latest.pt", a.out / "best.pt"
    best_val = float("inf")
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,
                                                ram_limit_percent=90))
    rng = random.Random(20260824)
    print(json.dumps({"event": "sft_lora_initialized", "examples": len(positions),
                      "total_parameters": total_params,
                      "trainable_parameters": trainable_params,
                      "loss_target": "assistant_response_only"},
                     ensure_ascii=False), flush=True)

    for step in range(1, a.steps + 1):
        item = read_example(a.data, positions[train[rng.randrange(len(train))]],
                            tok, a.max_len, bos, eos)
        if item is None:
            continue
        x, labels = item
        treinar_lora_voz.set_lora_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0), torch.zeros(1, 11))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels,
                               ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        if step % a.save_every == 0 or step == a.steps:
            val_loss = evaluate(model, a.data, [positions[i] for i in val],
                                tok, a.max_len, bos, eos)
            metrics = {"train_loss": float(loss.detach()), "val_loss": val_loss,
                       "dataset": "human_chats_sft_response_only",
                       "total_parameters": total_params,
                       "trainable_parameters": trainable_params,
                       "loss_target": "assistant_response_only",
                       "base": str(a.base)}
            save_checkpoint(latest, model=model, optimizer=optimizer,
                            step=step, metrics=metrics)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best, model=model, optimizer=optimizer,
                                step=step, metrics=metrics)
            print(json.dumps({"step": step, **metrics, "best_val": best_val,
                              **governor.throttle()}, ensure_ascii=False), flush=True)
    print(f"[OK] SFT LoRA concluído; melhor validação {best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()

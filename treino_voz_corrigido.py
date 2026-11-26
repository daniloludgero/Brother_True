"""Fine-tuning causal correto: a loss cobre somente a resposta.

Para seq = [BOS] + prompt + resposta + [EOS], o alvo é seq[1:].
Mascaramos apenas os primeiros len(prompt) alvos; assim o primeiro token
da resposta é aprendido a partir do último token do prompt.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .voz import Voz


def offsets(path: Path):
    result = []
    with path.open("rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                result.append(position)
    return result


def make_example(path, position, tok, max_len, pad, bos, eos):
    with path.open("rb") as handle:
        handle.seek(position)
        item = json.loads(handle.readline().decode("utf-8"))
    prompt = tok.encode(item["prompt"], add_special_tokens=False).ids
    response = tok.encode(item["response"], add_special_tokens=False).ids
    if not prompt or not response:
        return None
    prompt = prompt[-(max_len // 2):]
    response = response[: max_len - len(prompt) - 1]
    seq = [bos] + prompt + response + [eos]
    seq = seq[:max_len]
    x = torch.tensor(seq[:-1], dtype=torch.long)
    y = torch.tensor(seq[1:], dtype=torch.long)
    # x[0:len(prompt)] predicts prompt tokens; ignore those targets.
    y[:len(prompt)] = pad
    return x, y


@torch.no_grad()
def evaluate(model, path, positions, tok, max_len, pad, bos, eos):
    model.eval()
    values = []
    for position in positions[:40]:
        item = make_example(path, position, tok, max_len, pad, bos, eos)
        if item is None:
            continue
        x, y = item
        logits = model(x.unsqueeze(0), torch.zeros(1, 11),
                       tgt_key_padding_mask=(x.unsqueeze(0) == pad))
        values.append(float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y, ignore_index=pad)))
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()
    random.seed(20260823)
    torch.manual_seed(20260823)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(args.tokenizer))
    pad = tok.token_to_id("<pad>") or 0
    bos = tok.token_to_id("<bos>") or 1
    eos = tok.token_to_id("<eos>") or 2
    positions = offsets(args.data)
    order = list(range(len(positions)))
    random.shuffle(order)
    val = order[:max(1, len(order) // 10)]
    train = order[len(val):]
    model = Voz(vocab_size=tok.get_vocab_size(), d_model=384, nhead=6,
                num_layers=4, dim_feedforward=1536, max_len=args.max_len,
                pad_token_id=pad, bos_token_id=bos, eos_token_id=eos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    best = args.out / "best.pt"
    source = latest if latest.exists() else args.base
    state = load_checkpoint(source, model=model, optimizer=optimizer, map_location="cpu")
    step = int(state.get("step", 0))
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    best_val = float("inf")
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90))
    rng = random.Random(20260823 + step)
    rng.shuffle(train)
    cursor = 0
    while step < args.steps:
        if cursor >= len(train):
            rng.shuffle(train)
            cursor = 0
        item = make_example(args.data, positions[train[cursor]], tok,
                            args.max_len, pad, bos, eos)
        cursor += 1
        if item is None:
            continue
        x, y = item
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0), torch.zeros(1, 11),
                       tgt_key_padding_mask=(x.unsqueeze(0) == pad))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y,
                               ignore_index=pad)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % 25 == 0 or step == args.steps:
            val_loss = evaluate(model, args.data, [positions[i] for i in val],
                                tok, args.max_len, pad, bos, eos)
            metrics = {"train_loss": float(loss.detach()), "val_loss": val_loss,
                       "bos": bos, "eos": eos, "learning_rate": args.lr,
                       "loss_target": "response_only_aligned"}
            save_checkpoint(latest, model=model, optimizer=optimizer,
                            step=step, metrics=metrics)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best, model=model, optimizer=optimizer,
                                step=step, metrics=metrics)
            print(json.dumps({"step": step, **metrics, "best_val": best_val,
                              **governor.throttle()}, ensure_ascii=False), flush=True)
    print(f"[OK] Correção concluída; melhor validação {best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()

"""Treina a Voz 50M com sinais roteados de dois professores."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from . import treinar_lora_voz
    from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
    from .resource_governor import ResourceGovernor, ResourcePolicy
    from .treino_voz_texto_limpo import offsets
    from .voz import Voz
except (ImportError, ValueError):
    import treinar_lora_voz
    from checkpoint_manager_safe import load_checkpoint, save_checkpoint
    from resource_governor import ResourceGovernor, ResourcePolicy
    from treino_voz_texto_limpo import offsets
    from voz import Voz

treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


def expanded_base(path, old_vocab, new_vocab, tok, max_len):
    pad = tok.token_to_id("<pad>") or 0; bos = tok.token_to_id("<bos>") or 1; eos = tok.token_to_id("<eos>") or 2
    old = Voz(vocab_size=old_vocab, d_model=512, nhead=8, num_layers=8,
              dim_feedforward=2048, max_len=max_len, pad_token_id=pad,
              bos_token_id=bos, eos_token_id=eos)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_dict = payload.get("model", {}) if isinstance(payload, dict) else payload
    has_lora = any("lora" in k for k in model_dict.keys())
    if has_lora:
        treinar_lora_voz.add_lora(old, rank=8, alpha=16.0, dropout=0.05)
    load_checkpoint(path, model=old, map_location="cpu")
    new = Voz(vocab_size=new_vocab, d_model=512, nhead=8, num_layers=8,
              dim_feedforward=2048, max_len=max_len, pad_token_id=pad,
              bos_token_id=bos, eos_token_id=eos)
    if has_lora:
        treinar_lora_voz.add_lora(new, rank=8, alpha=16.0, dropout=0.05)
    state = new.state_dict()
    for key, value in old.state_dict().items():
        if key not in state:
            continue
        if state[key].shape == value.shape:
            state[key] = value
        elif key in ("token_embedding.weight", "lm_head.weight", "lm_head.bias", "lm_head.base.weight", "lm_head.base.bias"):
            state[key][:old_vocab] = value
    new.load_state_dict(state)
    return new


def read_example(path, pos, tok, max_len, bos, eos):
    with path.open("rb") as f:
        f.seek(pos); item = json.loads(f.readline().decode("utf-8"))
    prompt = tok.encode(item["prompt"], add_special_tokens=False).ids
    target = tok.encode(item["response"] + item.get("target_suffix", "<|im_end|><|end_of_text|>"), add_special_tokens=False).ids + [eos]
    target = target[:max_len - 2]
    prompt = prompt[-max(1, max_len - 1 - len(target)):]
    seq = [bos] + prompt + target
    x = torch.tensor(seq[:-1]); y = torch.tensor(seq[1:]); y[:len(prompt)] = -100
    return x, y


@torch.no_grad()
def evaluate(model, path, positions, tok, max_len, bos, eos):
    model.eval(); values = []
    for pos in positions[:32]:
        x, y = read_example(path, pos, tok, max_len, bos, eos)
        z = model(x.unsqueeze(0), torch.zeros(1, 11))
        values.append(float(F.cross_entropy(z.reshape(-1, z.size(-1)), y, ignore_index=-100)))
    return sum(values) / max(1, len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--save-every", type=int, default=20)
    a = p.parse_args()
    random.seed(20260825); torch.manual_seed(20260825); torch.set_num_threads(2)
    tok = Tokenizer.from_file(str(a.tokenizer)); positions = offsets(a.data)
    order = list(range(len(positions))); random.shuffle(order)
    val = order[:max(1, len(order)//5)]; train = order[len(val):]
    pad = tok.token_to_id("<pad>") or 0; bos = tok.token_to_id("<bos>") or 1; eos = tok.token_to_id("<eos>") or 2
    model = expanded_base(a.base, 16384, tok.get_vocab_size(), tok, a.max_len)
    total = sum(p.numel() for p in model.parameters())
    trainable = treinar_lora_voz.add_lora(model, a.rank, a.rank * 2.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr); a.out.mkdir(parents=True, exist_ok=True)
    gov = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90))
    best = float("inf"); rng = random.Random(20260825)
    print(json.dumps({"event":"initialized","examples":len(positions),"brief_complex_routed":True,
                      "total_parameters":total,"trainable_parameters":trainable},ensure_ascii=False),flush=True)
    for step in range(1, a.steps + 1):
        x, y = read_example(a.data, positions[train[rng.randrange(len(train))]], tok, a.max_len, bos, eos)
        treinar_lora_voz.set_lora_train_mode(model); opt.zero_grad(set_to_none=True)
        z = model(x.unsqueeze(0), torch.zeros(1, 11)); loss = F.cross_entropy(z.reshape(-1, z.size(-1)), y, ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % a.save_every == 0 or step == a.steps:
            val_loss = evaluate(model, a.data, [positions[i] for i in val], tok, a.max_len, bos, eos)
            metrics = {"train_loss":float(loss.detach()),"val_loss":val_loss,"step":step,
                       "total_parameters":total,"trainable_parameters":trainable,"dataset":"two_teacher_distillation"}
            save_checkpoint(a.out/"latest.pt",model=model,optimizer=opt,step=step,metrics=metrics)
            if val_loss < best: best = val_loss; save_checkpoint(a.out/"best.pt",model=model,optimizer=opt,step=step,metrics=metrics)
            print(json.dumps({"event":"checkpoint",**metrics,"best_val":best,**gov.throttle()},ensure_ascii=False),flush=True)
    print(json.dumps({"event":"complete","best_val":best,"out":str(a.out)},ensure_ascii=False),flush=True)


if __name__ == "__main__": main()

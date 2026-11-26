"""Piloto do pensamento com janela 256 e perda mascarada no prompt."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pensamento_linguistico import PensamentoLinguistico


def rows_from(path, limit):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("prompt") and r.get("think"):
                    rows.append(r)
                if len(rows) >= limit:
                    break
    if len(rows) < 16:
        raise RuntimeError(f"Exemplos insuficientes: {len(rows)}")
    return rows


def encode(tok, row, max_len):
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    prefix = list(tok.encode(row["prompt"] + "\n<think>\n").ids)
    target = list(tok.encode(row["think"]).ids)
    ids = ([bos] if bos is not None else []) + prefix + target + ([eos] if eos is not None else [])
    target_start = len(ids) - len(target) - (1 if eos is not None else 0)
    # Preserva a posição inicial do alvo; o limite atual é suficiente para a
    # maioria dos exemplos, e exemplos longos são truncados no final.
    ids = ids[: max_len + 1]
    target_start = min(target_start, len(ids))
    x, y = torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)
    # A loss não treina a repetição do prompt, apenas think e EOS.
    y[: max(0, target_start - 1)] = -100
    return x, y


def batch(tok, rows, max_len):
    pairs = [encode(tok, r, max_len) for r in rows]
    width = max(len(x) for x, _ in pairs)
    pad = tok.token_to_id("<pad>") or 0
    xs, ys = [], []
    for x, y in pairs:
        xs.append(F.pad(x, (0, width - len(x)), value=pad))
        ys.append(F.pad(y, (0, width - len(y)), value=-100))
    return torch.stack(xs).long(), torch.stack(ys).long()


def load_resized(base, vocab, max_len):
    model = PensamentoLinguistico(vocab_size=vocab, d_model=384, nhead=6,
                                  num_layers=4, dim_feedforward=1536, max_len=max_len)
    payload = torch.load(base, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model", payload))
    old_pos = state.pop("pos_embedding.weight", None)
    model.load_state_dict(state, strict=False)
    if old_pos is not None:
        with torch.no_grad():
            resized = F.interpolate(old_pos.T.unsqueeze(0), size=max_len,
                                    mode="linear", align_corners=False).squeeze(0).T
            model.pos_embedding.weight.copy_(resized)
    return model


def forward(model, x):
    return model(x, torch.zeros((x.size(0), 11), dtype=torch.float32))


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = forward(model, x)
    return float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=1024); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    random.seed(20260827); torch.manual_seed(20260827); torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = rows_from(args.data, args.rows); cut = max(8, int(len(rows) * .8))
    tok = Tokenizer.from_file(str(args.tokenizer)); model = load_resized(args.base, tok.get_vocab_size(), 256)
    train, val = rows[:cut], rows[cut:]; xv, yv = batch(tok, val, 256)
    before = evaluate(model, xv, yv)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    for step in range(1, args.steps + 1):
        start = ((step - 1) * 8) % len(train); chosen = [train[(start + j) % len(train)] for j in range(8)]
        x, y = batch(tok, chosen, 256); model.train(); opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, x).flatten(0, 1), y.flatten(), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step == args.steps or step % 25 == 0:
            print(f"step={step}/{args.steps} masked_train_loss={loss.item():.6f}", flush=True)
    after = evaluate(model, xv, yv)
    result = {"experiment": "pensamento_window256_masked_prompt", "rows": len(rows), "train_rows": len(train),
              "val_rows": len(val), "steps": args.steps, "context_window": 256, "loss_target": "think_plus_eos",
              "before_val_loss": before, "after_val_loss": after, "delta_after_minus_before": after - before,
              "productive_by_masked_loss": after < before, "teacher_approved": False,
              "production_checkpoint_untouched": True}
    torch.save({"model_state_dict": model.state_dict(), "metrics": result}, args.out / "pilot.pt")
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

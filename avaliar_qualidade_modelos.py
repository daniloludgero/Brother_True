"""Compara o encoder congelado com a Voz geradora em tarefas compatíveis.

As métricas são separadas porque o MiniLM é um encoder sem cabeça geradora,
enquanto a Voz é um decoder causal. Não tratamos similaridade como resposta
gerada; medimos alinhamento semântico no primeiro e geração condicional no
segundo.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

from .checkpoint_manager_safe import load_checkpoint
from .voz import Voz


def load_records(path: Path, limit: int):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            item = json.loads(line)
            if item.get("prompt") and item.get("response"):
                records.append(item)
    return records


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def voz_generate(model, tok, prompt, max_new_tokens=48):
    bos = model.bos_token_id
    eos = model.eos_token_id
    ids = tok.encode(prompt, add_special_tokens=False).ids[-(model.max_len - 1):]
    prefix = [bos] + ids
    sequence = list(prefix)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([sequence[-model.max_len:]], dtype=torch.long)
            logits = model(x, torch.zeros(1, 11))[0, -1]
            next_id = int(torch.argmax(logits).item())
            sequence.append(next_id)
            if next_id == eos or len(sequence) >= model.max_len:
                break
    generated = sequence[len(prefix):]
    return tok.decode(generated, skip_special_tokens=True), generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--voz", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    records = load_records(args.data, args.limit)
    prompts = [item["prompt"] for item in records]
    responses = [item["response"] for item in records]

    frozen = SentenceTransformer(str(args.minilm), local_files_only=True)
    frozen.eval()
    with torch.no_grad():
        prompt_vec = frozen.encode(prompts, batch_size=16, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        response_vec = frozen.encode(responses, batch_size=16, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
    scores = prompt_vec @ response_vec.T
    positive = scores.diagonal()
    negative = scores[range(len(records)), [(i + 1) % len(records) for i in range(len(records))]]
    ranks = []
    for i in range(len(records)):
        order = scores[i].argsort()[::-1]
        ranks.append(int((order == i).nonzero()[0][0]) + 1)
    frozen_report = {
        "parameters": sum(p.numel() for p in frozen.parameters()),
        "examples": len(records),
        "positive_cosine_mean": float(positive.mean()),
        "negative_shifted_cosine_mean": float(negative.mean()),
        "pairwise_positive_beats_negative": float((positive > negative).mean()),
        "retrieval_at_1": float((torch.tensor(ranks) == 1).float().mean()),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(ranks)),
        "median_rank": float(sorted(ranks)[len(ranks) // 2]),
    }

    tok = Tokenizer.from_file(str(args.tokenizer))
    voz = Voz(vocab_size=tok.get_vocab_size(), d_model=384, nhead=6,
              num_layers=4, dim_feedforward=1536, max_len=128)
    state = load_checkpoint(args.voz, model=voz, map_location="cpu")
    voz.eval()
    generations = []
    first_eos = 0
    exact = 0
    for item in records:
        text, ids = voz_generate(voz, tok, item["prompt"])
        if ids and ids[0] == voz.eos_token_id:
            first_eos += 1
        if normalize(text) == normalize(item["response"]):
            exact += 1
        generations.append({"prompt": item["prompt"], "expected": item["response"],
                            "generated": text, "token_count": len(ids)})
    current_report = {
        "parameters": sum(p.numel() for p in voz.parameters()),
        "checkpoint_step": state["step"],
        "checkpoint_metrics": state["metrics"],
        "examples": len(records),
        "first_token_eos_rate": first_eos / len(records),
        "empty_generation_rate": sum(not x["generated"] for x in generations) / len(records),
        "exact_match_rate": exact / len(records),
        "unique_generation_rate": len({normalize(x["generated"]) for x in generations}) / len(records),
        "mean_generated_tokens": sum(x["token_count"] for x in generations) / len(records),
        "samples": generations[:5],
    }
    print(json.dumps({"frozen_encoder": frozen_report, "voz_current": current_report},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

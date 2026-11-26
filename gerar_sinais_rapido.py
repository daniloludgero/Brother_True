"""Versão econômica da geração de sinais: encoder congelado com 64 tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("dados_preparados/dataset_normalizado.jsonl"))
    p.add_argument("--output", type=Path, default=Path("dados_preparados/sinais_treinamento.jsonl"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer_voz_16k.json"))
    p.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer))
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", cache_folder=str(args.cache_dir))
    enc.max_seq_length = 64
    rng = np.random.default_rng(20260823)
    proj = rng.normal(0.0, 1.0 / np.sqrt(384), size=(384, 11)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    batch = []
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as out:
        for line in src:
            item = json.loads(line)
            if item.get("text", "").strip():
                batch.append(item)
            if len(batch) >= args.batch_size:
                count += write_batch(batch, enc, tok, proj, out, args.max_tokens)
                batch.clear()
        if batch:
            count += write_batch(batch, enc, tok, proj, out, args.max_tokens)
    print(json.dumps({"examples": count, "max_semantic_tokens": 64}, ensure_ascii=False))


def write_batch(batch, enc, tok, proj, out, max_tokens):
    texts = [x["text"][:2048] for x in batch]
    embeddings = enc.encode(texts, batch_size=len(batch), convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
    n = 0
    for item, emb in zip(batch, embeddings):
        ids = tok.encode(item["text"]).ids[:max_tokens]
        if len(ids) < 2:
            continue
        record = {"input_ids": ids[:-1], "target_ids": ids[1:],
                  "humor": np.tanh(emb @ proj).astype(np.float32).round(6).tolist(),
                  "source": item["source"], "file": item["file"], "row": item["row"]}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        n += 1
    return n


if __name__ == "__main__":
    main()

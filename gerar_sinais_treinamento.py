"""Gera sinais locais de treino usando o encoder semântico congelado.

Saída: JSONL com input_ids, target_ids e humor_semantico (11 dimensões).
O encoder não recebe gradiente; a projeção é fixa e reprodutível.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer


def _humor_semantico(embedding: np.ndarray, projection: np.ndarray) -> list[float]:
    values = np.tanh(embedding @ projection)
    return values.astype(np.float32).round(6).tolist()


def gerar(input_jsonl: Path, output_jsonl: Path, tokenizer_path: Path, model_name: str,
          cache_dir: Path, batch_size: int, max_tokens: int) -> int:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoder = SentenceTransformer(model_name, device="cpu", cache_folder=str(cache_dir))
    rng = np.random.default_rng(20260823)
    projection = rng.normal(0.0, 1.0 / np.sqrt(384), size=(384, 11)).astype(np.float32)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_jsonl.open("r", encoding="utf-8") as source, output_jsonl.open("w", encoding="utf-8") as target:
        batch_records: list[dict] = []
        for line in source:
            record = json.loads(line)
            text = record.get("text", "").strip()
            if text:
                batch_records.append(record)
            if len(batch_records) < batch_size:
                continue
            count += _write_batch(batch_records, encoder, tokenizer, projection, target, max_tokens)
            batch_records.clear()
        if batch_records:
            count += _write_batch(batch_records, encoder, tokenizer, projection, target, max_tokens)
    return count


def _write_batch(records, encoder, tokenizer, projection, target, max_tokens) -> int:
    embeddings = encoder.encode([r["text"] for r in records], batch_size=len(records), convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
    written = 0
    for record, embedding in zip(records, embeddings):
        ids = tokenizer.encode(record["text"]).ids[:max_tokens]
        if len(ids) < 2:
            continue
        item = {
            "input_ids": ids[:-1],
            "target_ids": ids[1:],
            "humor": _humor_semantico(embedding, projection),
            "source": record["source"],
            "file": record["file"],
            "row": record["row"],
        }
        target.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dados_preparados/dataset_normalizado.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("dados_preparados/sinais_treinamento.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer_voz_16k.json"))
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    count = gerar(args.input, args.output, args.tokenizer, args.model, args.cache_dir, args.batch_size, args.max_tokens)
    print(json.dumps({"examples": count, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

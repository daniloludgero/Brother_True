"""Constrói a memória RAG com MiniLM congelado e somente dados locais."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .resource_governor import ResourceGovernor, ResourcePolicy


def retrieval_text(prompt: str) -> str:
    text = re.sub(r"</?(?:user|assistant)>", " ", prompt, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            query_text = retrieval_text(item.get("prompt", ""))
            response = item.get("response", "").strip()
            if query_text and response:
                records.append({
                    "source": item.get("source"),
                    "row": item.get("row"),
                    "query": query_text,
                    "response": response,
                })
    model = SentenceTransformer(str(args.minilm), device="cpu",
                                local_files_only=True)
    model.eval()
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,
                                                ram_limit_percent=90,
                                                sleep_seconds=1.0))
    embeddings = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        embeddings.append(model.encode(
            [item["query"] for item in batch],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False))
        if (start // args.batch_size + 1) % 25 == 0 or start + args.batch_size >= len(records):
            status = governor.throttle()
            print(json.dumps({"processed": min(start + args.batch_size, len(records)),
                              "total": len(records), **status}, ensure_ascii=False),
                  flush=True)
    matrix = np.concatenate(embeddings, axis=0)
    np.save(args.out / "embeddings.npy", matrix)
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "model": "all-MiniLM-L6-v2",
        "model_path": str(args.minilm),
        "dimension": int(matrix.shape[1]),
        "records": len(records),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "reasoning_indexed": False,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

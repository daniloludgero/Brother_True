"""Vetoriza todos os registros do GraphRAG rápido sem reconstruir o grafo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .resource_governor import ResourceGovernor, ResourcePolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.graph / "records.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    torch.set_num_threads(2)
    model = SentenceTransformer(str(args.minilm), device="cpu", local_files_only=True)
    model.eval()
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=90))
    chunks = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        chunks.append(model.encode([item["query"] for item in batch],
                                   batch_size=args.batch_size,
                                   normalize_embeddings=True,
                                   convert_to_numpy=True,
                                   show_progress_bar=False).astype(np.float32, copy=False))
        if (start // args.batch_size + 1) % 16 == 0 or start + args.batch_size >= len(records):
            print(json.dumps({"processed":min(start+args.batch_size,len(records)),
                              "total":len(records),**governor.throttle()},ensure_ascii=False),flush=True)
    matrix=np.concatenate(chunks,axis=0)
    np.save(args.out / "embeddings.npy", matrix)
    manifest={"type":"semantic_index","records":len(records),"dimension":int(matrix.shape[1]),
              "dtype":str(matrix.dtype),"normalized":True,"base_graph":str(args.graph)}
    (args.out / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__": main()

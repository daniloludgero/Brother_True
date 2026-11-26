"""Índice RAG local baseado no MiniLM congelado."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalRAG:
    def __init__(self, index_dir: str | Path, minilm_path: str | Path,
                 device: str = "cpu"):
        root = Path(index_dir)
        self.embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
        with (root / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.encoder = SentenceTransformer(str(minilm_path), device=device,
                                            local_files_only=True)
        self.encoder.eval()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self.encoder.encode([query], normalize_embeddings=True,
                                     convert_to_numpy=True,
                                     show_progress_bar=False)[0]
        scores = self.embeddings @ vector
        count = min(top_k, len(scores))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = indices[np.argsort(-scores[indices])]
        return [{**self.records[int(index)], "score": float(scores[int(index)])}
                for index in indices]

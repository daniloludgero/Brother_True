"""Consulta GraphRAG local com busca semântica e expansão por tópicos."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class GraphRAGLocal:
    def __init__(self, graph_dir: str | Path, minilm_path: str | Path):
        root = Path(graph_dir)
        self.embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
        with (root / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.db = sqlite3.connect(root / "graph.sqlite3")
        self.encoder = SentenceTransformer(str(minilm_path), device="cpu",
                                            local_files_only=True)
        self.encoder.eval()

    def _topic_neighbors(self, record_indices: list[int]) -> set[int]:
        topics = set()
        for index in record_indices:
            rows = self.db.execute(
                "SELECT dst FROM edges WHERE src=? AND relation='has_topic'",
                (f"item:{index}",),
            ).fetchall()
            topics.update(row[0] for row in rows)
        if not topics:
            return set()
        placeholders = ",".join("?" for _ in topics)
        rows = self.db.execute(
            f"SELECT DISTINCT src FROM edges WHERE dst IN ({placeholders}) AND relation='has_topic' LIMIT 400",
            tuple(topics),
        ).fetchall()
        result = set()
        for (node_id,) in rows:
            if node_id.startswith("item:"):
                result.add(int(node_id.split(":", 1)[1]))
        return result

    def search(self, query: str, top_k: int = 5, seed_k: int = 24) -> list[dict]:
        vector = self.encoder.encode([query], normalize_embeddings=True,
                                     convert_to_numpy=True,
                                     show_progress_bar=False)[0]
        scores = self.embeddings @ vector
        seed_k = min(seed_k, len(scores))
        seed_indices = np.argpartition(-scores, seed_k - 1)[:seed_k]
        candidates = set(int(index) for index in seed_indices)
        candidates.update(self._topic_neighbors(list(candidates)))
        ranked = []
        seed_set = set(int(index) for index in seed_indices)
        for index in candidates:
            semantic = float(scores[index])
            shared = 1 if index not in seed_set else 0
            final_score = semantic + 0.005 * shared
            ranked.append((final_score, index, semantic, shared))
        ranked.sort(reverse=True)
        result = []
        for final_score, index, semantic, shared in ranked[:top_k]:
            item = dict(self.records[index])
            item.update({"score": final_score, "semantic_score": semantic,
                         "graph_expansion": bool(shared),
                         "record_index": index})
            result.append(item)
        return result

    def context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)
        blocks = []
        for result in results:
            blocks.append(
                f"[fonte={result['source']} linha={result['row']} score={result['score']:.3f}]\n"
                f"Pergunta relacionada: {result['query']}\n"
                f"Resposta registrada: {result['response']}"
            )
        return "\n\n".join(blocks)

    def close(self):
        self.db.close()

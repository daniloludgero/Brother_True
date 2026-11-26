"""GraphRAG rápido: grafo lexical/tópico + reranking MiniLM congelado."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]{4,}")
STOPWORDS = {"about","after","also","com","como","from","have","into","mais","para","que","this","that","the","they","what","when","where","with","your","user","assistant","please"}


class GraphRAGRapido:
    def __init__(self, graph_dir: str | Path, minilm_path: str | Path):
        root = Path(graph_dir)
        self.db = sqlite3.connect(root / "graph.sqlite3")
        with (root / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.encoder = SentenceTransformer(str(minilm_path), device="cpu", local_files_only=True)
        self.encoder.eval()

    def _terms(self, query: str) -> list[str]:
        return list(dict.fromkeys(t for t in TOKEN_RE.findall(query.lower()) if t not in STOPWORDS))[:16]

    def search(self, query: str, top_k: int = 5, candidate_limit: int = 256) -> list[dict]:
        query_terms = self._terms(query)
        candidates = set()
        if query_terms:
            placeholders = ",".join("?" for _ in query_terms)
            rows = self.db.execute(
                f"SELECT record_index FROM term_index WHERE term IN ({placeholders}) LIMIT ?",
                (*query_terms, candidate_limit),
            ).fetchall()
            candidates.update(int(row[0]) for row in rows)
        if not candidates:
            return []
        candidate_ids = sorted(candidates)
        texts = [self.records[index]["query"] for index in candidate_ids]
        vectors = self.encoder.encode([query] + texts, batch_size=32,
                                      normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False)
        scores = vectors[1:] @ vectors[0]
        ranked = []
        for index, semantic in zip(candidate_ids, scores):
            overlap = len(set(query_terms) & set(self._terms(self.records[index]["query"])))
            score = float(semantic) + min(overlap, 4) * 0.005
            item = dict(self.records[index])
            item.update({"score": score, "semantic_score": float(semantic),
                         "topic_overlap": overlap, "record_index": index})
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def close(self):
        self.db.close()

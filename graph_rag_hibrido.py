"""GraphRAG híbrido: busca semântica global + grafo de tópicos local."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class GraphRAGHibrido:
    def __init__(self, graph_dir: str | Path, semantic_dir: str | Path, minilm_path: str | Path):
        graph=Path(graph_dir); sem=Path(semantic_dir)
        self.records=[json.loads(line) for line in (graph/"records.jsonl").open(encoding="utf-8") if line.strip()]
        self.embeddings=np.load(sem/"embeddings.npy",mmap_mode="r")
        self.db=sqlite3.connect(graph/"graph.sqlite3")
        self.encoder=SentenceTransformer(str(minilm_path),device="cpu",local_files_only=True); self.encoder.eval()

    def _neighbors(self, indices):
        topics=set()
        for index in indices:
            topics.update(row[0] for row in self.db.execute("SELECT dst FROM edges WHERE src=? AND relation='has_topic'",(f"item:{index}",)).fetchall())
        if not topics: return set()
        marks=','.join('?' for _ in topics)
        rows=self.db.execute(f"SELECT DISTINCT src FROM edges WHERE dst IN ({marks}) AND relation='has_topic' LIMIT 400",tuple(topics)).fetchall()
        return {int(node.split(':',1)[1]) for (node,) in rows if node.startswith('item:')}

    def search(self, query: str, top_k: int=5, seed_k: int=32):
        vector=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
        scores=self.embeddings @ vector; seed_k=min(seed_k,len(scores))
        seeds=np.argpartition(-scores,seed_k-1)[:seed_k]; candidates=set(int(x) for x in seeds); candidates.update(self._neighbors(candidates))
        seed_set=set(int(x) for x in seeds); ranked=[]
        for index in candidates:
            semantic=float(scores[index]); graph_bonus=0.005 if index not in seed_set else 0.0
            item=dict(self.records[index]); item.update({'score':semantic+graph_bonus,'semantic_score':semantic,'graph_expansion':index not in seed_set,'topic_overlap':0,'record_index':index})
            ranked.append(item)
        ranked.sort(key=lambda item:item['score'],reverse=True)
        return ranked[:top_k]

    def close(self): self.db.close()

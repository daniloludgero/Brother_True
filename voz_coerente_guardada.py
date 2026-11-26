"""Política conservadora para a primeira Voz GraphRAG."""

from __future__ import annotations

from pathlib import Path

from .graph_rag_rapido import GraphRAGRapido


class VozCoerenteGuardada:
    """Só entrega recuperação quando evidência e margem são suficientes."""

    def __init__(self, graph_dir: str | Path, minilm_path: str | Path,
                 min_score: float = 0.72, min_margin: float = 0.05,
                 min_topic_overlap: int = 2):
        self.rag = GraphRAGRapido(graph_dir, minilm_path)
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_topic_overlap = min_topic_overlap

    def responder(self, query: str, top_k: int = 5) -> dict:
        results = self.rag.search(query, top_k=top_k)
        if not results:
            return {"mode": "guarded_fallback", "answer": "Não encontrei contexto suficiente para responder com segurança.", "results": []}
        best = results[0]
        margin = best["score"] - (results[1]["score"] if len(results) > 1 else 0.0)
        accepted = (
            best["score"] >= self.min_score
            and margin >= self.min_margin
            and best["topic_overlap"] >= self.min_topic_overlap
        )
        if accepted:
            return {"mode": "graph_rag_grounded", "answer": best["response"],
                    "confidence": best["score"], "margin": margin,
                    "topic_overlap": best["topic_overlap"],
                    "source": best["source"], "row": best["row"],
                    "results": results}
        return {"mode": "guarded_fallback", "answer": "Encontrei referências próximas, mas não há evidência suficiente para responder automaticamente.",
                "confidence": best["score"], "margin": margin,
                "topic_overlap": best["topic_overlap"], "results": results}

    def close(self):
        self.rag.close()

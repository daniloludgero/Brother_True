"""Política GraphRAG calibrada: recupera paráfrases fortes, bloqueia ruído."""

from __future__ import annotations

from pathlib import Path

from .graph_rag_rapido import GraphRAGRapido


class VozCoerenteCalibrada:
    def __init__(self, graph_dir: str | Path, minilm_path: str | Path):
        self.rag = GraphRAGRapido(graph_dir, minilm_path)

    def responder(self, query: str, top_k: int = 5) -> dict:
        results = self.rag.search(query, top_k=top_k)
        if not results:
            return {"mode":"guarded_fallback", "answer":"Não encontrei contexto suficiente para responder com segurança.", "results":[]}
        best=results[0]; second=results[1]["score"] if len(results)>1 else 0.0
        margin=best["score"]-second; overlap=best["topic_overlap"]
        strict=best["score"]>=0.72 and margin>=0.05 and overlap>=2
        semantic=best["score"]>=0.65 and margin>=0.05 and overlap>=5
        if strict or semantic:
            return {"mode":"graph_rag_grounded", "answer":best["response"],
                    "confidence":best["score"], "margin":margin,
                    "topic_overlap":overlap, "source":best["source"],
                    "row":best["row"], "results":results}
        return {"mode":"guarded_fallback", "answer":"Encontrei referências próximas, mas não há evidência suficiente para responder automaticamente.",
                "confidence":best["score"], "margin":margin,
                "topic_overlap":overlap, "results":results}

    def close(self): self.rag.close()

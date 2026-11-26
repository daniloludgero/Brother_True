"""Camada inicial de Voz: resposta fundamentada antes da geração neural."""

from __future__ import annotations

from pathlib import Path

from .graph_rag_local import GraphRAGLocal


class VozCoerente:
    """Prioriza respostas recuperadas e rastreáveis.

    O decoder atual ainda pode emitir EOS vazio. Por isso, a política inicial
    não entrega saída neural de baixa confiança ao usuário.
    """

    def __init__(self, graph_dir: str | Path, minilm_path: str | Path,
                 min_score: float = 0.55, min_margin: float = 0.015):
        self.rag = GraphRAGLocal(graph_dir, minilm_path)
        self.min_score = min_score
        self.min_margin = min_margin

    def responder(self, query: str, top_k: int = 5) -> dict:
        results = self.rag.search(query, top_k=top_k)
        if not results:
            return {"answer": "Não encontrei contexto suficiente para responder com segurança.",
                    "mode": "guarded_fallback", "results": []}
        best = results[0]
        margin = best["score"] - (results[1]["score"] if len(results) > 1 else 0.0)
        if best["score"] >= self.min_score and margin >= self.min_margin:
            return {"answer": best["response"], "mode": "graph_rag_grounded",
                    "confidence": best["score"], "margin": margin,
                    "source": best["source"], "row": best["row"],
                    "results": results}
        return {"answer": "Encontrei referências próximas, mas não há confiança suficiente para uma resposta automática.",
                "mode": "guarded_fallback", "confidence": best["score"],
                "margin": margin, "results": results}

    def close(self):
        self.rag.close()

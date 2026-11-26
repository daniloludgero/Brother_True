"""Camada guardada sobre o GraphRAG híbrido global."""

from __future__ import annotations

from pathlib import Path

from .graph_rag_hibrido import GraphRAGHibrido


class VozCoerenteHibrida:
    def __init__(self, graph_dir: str | Path, semantic_dir: str | Path, minilm_path: str | Path,
                 min_score: float = 0.72, min_margin: float = 0.05):
        self.rag=GraphRAGHibrido(graph_dir,semantic_dir,minilm_path); self.min_score=min_score; self.min_margin=min_margin

    def responder(self, query: str, top_k: int=5):
        results=self.rag.search(query,top_k=top_k)
        if not results: return {'mode':'guarded_fallback','answer':'Não encontrei contexto suficiente para responder com segurança.','results':[]}
        best=results[0]; margin=best['score']-(results[1]['score'] if len(results)>1 else 0.0)
        if best['score']>=self.min_score and margin>=self.min_margin:
            return {'mode':'graph_rag_grounded','answer':best['response'],'confidence':best['score'],'margin':margin,'source':best['source'],'row':best['row'],'results':results}
        return {'mode':'guarded_fallback','answer':'Encontrei referências próximas, mas não há evidência suficiente para responder automaticamente.','confidence':best['score'],'margin':margin,'results':results}

    def close(self): self.rag.close()

"""Endpoint local: GraphRAG direto, Qwen agentic com retry e fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .gerar_agentic_com_retentativa import gerar_agentic
from .graph_rag_rapido import GraphRAGRapido
from .treinar_professor_agentic_lora import add_lora
from .voz_coerente_guardada import VozCoerenteGuardada


def carregar_qwen(base: Path, adapter: Path | None = None):
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
                                                  dtype=torch.float32, low_cpu_mem_usage=True)
    if adapter:
        add_lora(model, rank=4, alpha=8.0)
        payload = torch.load(adapter, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("adapter", payload.get("adapter_state_dict", {})), strict=False)
    model.eval()
    return model, tokenizer


class EndpointContextualSeguro:
    def __init__(self, model, tokenizer, rag=None, *, deepseek_available: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.rag = rag
        self.deepseek_available = deepseek_available

    def responder(self, query: str) -> dict[str, Any]:
        rag_result = self.rag.responder(query, top_k=5) if self.rag else None
        # Evidência aprovada não passa por uma segunda interpretação generativa.
        if rag_result and rag_result.get("mode") == "graph_rag_grounded":
            return {"route": "graphrag_direct", "answer": rag_result.get("answer", ""),
                    "source": rag_result.get("source"), "confidence": rag_result.get("confidence"),
                    "margin": rag_result.get("margin"), "rag": rag_result}
        generated = gerar_agentic(self.model, self.tokenizer, query,
                                  rag_result=rag_result,
                                  deepseek_available=self.deepseek_available)
        return {"route": generated["routing"]["route"], "generation": generated,
                "rag": rag_result}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--graph", type=Path, default=None)
    p.add_argument("--minilm", type=Path, default=None)
    a = p.parse_args()
    torch.set_num_threads(2)
    model, tokenizer = carregar_qwen(a.base, a.adapter)
    rag = VozCoerenteGuardada(a.graph, a.minilm) if a.graph and a.minilm else None
    try:
        endpoint = EndpointContextualSeguro(model, tokenizer, rag)
        print(json.dumps(endpoint.responder(a.query), ensure_ascii=False, indent=2), flush=True)
    finally:
        if rag:
            rag.close()


if __name__ == "__main__":
    main()

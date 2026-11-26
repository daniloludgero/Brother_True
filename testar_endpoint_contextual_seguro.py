"""Teste estrutural do endpoint sem carregar o Qwen."""
from __future__ import annotations

from .endpoint_contextual_seguro import EndpointContextualSeguro


class FakeRag:
    def __init__(self, result):
        self.result = result

    def responder(self, query, top_k=5):
        return self.result


def main():
    endpoint = EndpointContextualSeguro(None, None, FakeRag({
        "mode": "graph_rag_grounded", "answer": "fonte aprovada", "source": "local"
    }))
    result = endpoint.responder("pergunta")
    assert result["route"] == "graphrag_direct"
    assert result["answer"] == "fonte aprovada"
    print("contextual_endpoint_tests=2 passed")


if __name__ == "__main__":
    main()

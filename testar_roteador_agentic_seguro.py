"""Teste rápido do contrato de validação e fallback agentic."""
from __future__ import annotations

from .roteador_agentic_seguro import validar_e_rotear


def main() -> None:
    valid = validar_e_rotear('{"final_answer":"ok","tool_calls":[{"name":"verify","arguments":{}}]}')
    assert valid["validation"]["valid"] and valid["routing"]["route"] == "qwen"

    invalid = validar_e_rotear('{"final_answer":"ok","tool_calls":[{"arguments":{}}]}')
    assert not invalid["validation"]["valid"] and invalid["routing"]["route"] == "manual_review"

    grounded = validar_e_rotear("texto inválido", rag_result={
        "mode": "graph_rag_grounded", "answer": "resposta com fonte"
    })
    assert grounded["routing"]["route"] == "graphrag_direct"
    print("agentic_router_tests=3 passed")


if __name__ == "__main__":
    main()

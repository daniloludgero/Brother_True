"""Validação estrita e roteamento seguro para respostas agentic do Qwen."""
from __future__ import annotations

import json
from typing import Any


def validar_json_agentic(texto: str) -> dict[str, Any]:
    """Aceita apenas JSON canônico; não tenta consertar texto arbitrário."""
    try:
        value = json.loads(texto.strip())
    except (TypeError, json.JSONDecodeError) as exc:
        return {"valid": False, "reason": "json_invalid", "error": str(exc)}
    if not isinstance(value, dict):
        return {"valid": False, "reason": "root_not_object"}
    final = value.get("final_answer")
    calls = value.get("tool_calls")
    if not isinstance(final, str):
        return {"valid": False, "reason": "final_answer_not_string"}
    if not isinstance(calls, list):
        return {"valid": False, "reason": "tool_calls_not_list"}
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            return {"valid": False, "reason": "tool_call_not_object", "index": index}
        if not isinstance(call.get("name"), str) or not call["name"].strip():
            return {"valid": False, "reason": "tool_name_missing", "index": index}
        if not isinstance(call.get("arguments"), dict):
            return {"valid": False, "reason": "tool_arguments_not_object", "index": index}
    if not final.strip() and not calls:
        return {"valid": False, "reason": "empty_response"}
    return {"valid": True, "value": {"final_answer": final.strip(), "tool_calls": calls}}


def prompt_de_retentativa(prompt: str, erro: str) -> str:
    return (f"{prompt}\n\nA saída anterior falhou na validação ({erro}). "
            "Responda somente com um objeto JSON válido, sem Markdown, exatamente no formato "
            '{"final_answer":"...","tool_calls":[{"name":"tool","arguments":{}}]}.')


def rotear_falha(validation: dict[str, Any], *, rag_result: dict[str, Any] | None = None,
                 deepseek_available: bool = False) -> dict[str, Any]:
    if validation.get("valid"):
        return {"route": "qwen", "payload": validation["value"]}
    if rag_result and rag_result.get("mode") == "graph_rag_grounded":
        return {"route": "graphrag_direct", "answer": rag_result.get("answer", ""),
                "reason": validation.get("reason")}
    if deepseek_available:
        return {"route": "deepseek_review", "reason": validation.get("reason")}
    return {"route": "manual_review", "reason": validation.get("reason"),
            "deepseek_required": True}


def validar_e_rotear(texto: str, *, rag_result: dict[str, Any] | None = None,
                     deepseek_available: bool = False) -> dict[str, Any]:
    validation = validar_json_agentic(texto)
    return {"validation": validation,
            "routing": rotear_falha(validation, rag_result=rag_result,
                                     deepseek_available=deepseek_available)}

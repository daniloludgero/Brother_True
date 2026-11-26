"""Geração Qwen com validação estrita, retry e fallback explícito."""
from __future__ import annotations

from typing import Any

import torch

from .roteador_agentic_seguro import prompt_de_retentativa, validar_json_agentic, rotear_falha


@torch.no_grad()
def gerar_agentic(model, tokenizer, prompt: str, *, max_len: int = 256,
                  max_new: int = 96, retries: int = 2,
                  rag_result: dict[str, Any] | None = None,
                  deepseek_available: bool = False) -> dict[str, Any]:
    """Gera no máximo retries+1 vezes; nunca promove texto inválido a resposta."""
    attempts = []
    current_prompt = prompt
    device = next(model.parameters()).device
    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        for attempt in range(retries + 1):
            full_prompt = current_prompt + "\n\nASSISTANT_JSON:\n"
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True,
                               max_length=max_len - max_new)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:],
                                   skip_special_tokens=True).strip()
            validation = validar_json_agentic(raw)
            attempts.append({"attempt": attempt + 1, "raw": raw,
                             "validation": validation})
            if validation.get("valid"):
                return {"status": "accepted", "attempts": attempts,
                        "routing": rotear_falha(validation, rag_result=rag_result,
                                                 deepseek_available=deepseek_available)}
            current_prompt = prompt_de_retentativa(prompt, validation.get("reason", "invalid"))
    finally:
        tokenizer.truncation_side = original_side
    last = attempts[-1]["validation"] if attempts else {"valid": False, "reason": "no_attempt"}
    return {"status": "rejected", "attempts": attempts,
            "routing": rotear_falha(last, rag_result=rag_result,
                                    deepseek_available=deepseek_available)}

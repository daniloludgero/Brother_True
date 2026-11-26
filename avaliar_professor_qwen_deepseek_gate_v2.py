"""Gate corrigido: preserva o fim do prompt e o marcador ASSISTANT_JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .avaliar_professor_qwen_deepseek_gate import reserved_rows
from .normalizar_saida_professor_agentic_v2 import normalize
from .treinar_professor_agentic_lora import add_lora, compact_target, encode


@torch.no_grad()
def evaluate(model, tok, rows, max_len: int, max_new: int) -> dict:
    losses, structural, examples = [], [], []
    original_side = tok.truncation_side
    tok.truncation_side = "left"
    try:
        for row in rows:
            x, y = encode(row, tok, max_len)
            logits = model(input_ids=x.unsqueeze(0)).logits
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y, ignore_index=-100)))
            prompt = row["prompt"] + "\n\nASSISTANT_JSON:\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len - max_new)
            generated = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            normalized = normalize(text)
            try:
                reference = json.loads(row["target"])
            except json.JSONDecodeError:
                reference = {}
            ref_calls = {c.get("name") for c in reference.get("tool_calls", []) if isinstance(c, dict)}
            out_calls = {c.get("name") for c in normalized.get("tool_calls", []) if isinstance(c, dict)}
            structural.append({"final_answer_present": bool(normalized.get("final_answer", "").strip()),
                               "tool_schema_present": bool(normalized.get("tool_calls")),
                               "tool_name_overlap": bool(ref_calls & out_calls) if ref_calls else not out_calls})
            if len(examples) < 8:
                examples.append({"kind": row.get("kind", ""), "reference": reference,
                                 "generated": text, "normalized": normalized})
    finally:
        tok.truncation_side = original_side
    loss = sum(losses) / max(1, len(losses))
    return {"samples": len(rows), "loss": loss, "perplexity": math.exp(min(loss, 20.0)),
            "final_answer_present_rate": sum(x["final_answer_present"] for x in structural) / max(1, len(structural)),
            "tool_schema_present_rate": sum(x["tool_schema_present"] for x in structural) / max(1, len(structural)),
            "tool_name_overlap_rate": sum(x["tool_name_overlap"] for x in structural) / max(1, len(structural)),
            "examples": examples}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True); p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    p.add_argument("--skip", type=int, default=256); p.add_argument("--rows", type=int, default=4)
    p.add_argument("--max-len", type=int, default=256); p.add_argument("--max-new", type=int, default=96)
    a = p.parse_args(); torch.set_num_threads(2)
    rows = reserved_rows(a.parquet, a.skip, a.rows)
    tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.teacher), local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True)
    add_lora(model, rank=4, alpha=8.0)
    payload = torch.load(a.adapter, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("adapter", payload.get("adapter_state_dict", {})), strict=False); model.eval()
    metrics = evaluate(model, tok, rows, a.max_len, a.max_new)
    result = {"evaluation": "corrected frozen Qwen gate", "truncation": "left_preserving_assistant_marker",
              "adapter": str(a.adapter), "parquet": str(a.parquet), "reserved_skip": a.skip, "metrics": metrics}
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": {k: v for k, v in metrics.items() if k != "examples"}, "report": str(a.report)}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

"""Variante CPU do gerador de sinais, usando Qwen em float32."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MESSAGE_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)


def get_messages(prompt: str):
    return [{"role": role, "content": text.strip()}
            for role, text in MESSAGE_RE.findall(prompt) if text.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=192)
    p.add_argument("--max-new-tokens", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(2)
    random.seed(20260824)
    rows = []
    with a.data.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if get_messages(row.get("prompt", "")):
                rows.append(row)
    random.shuffle(rows)
    rows = rows[:a.limit]
    tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(a.teacher), local_files_only=True, dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with a.output.open("w", encoding="utf-8") as out:
        for seen, row in enumerate(rows, 1):
            try:
                enc = tok.apply_chat_template(
                    get_messages(row["prompt"]), add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt")
                ids = enc["input_ids"][:, -a.max_input_tokens:]
                mask = enc["attention_mask"][:, -a.max_input_tokens:]
                with torch.inference_mode():
                    gen = model.generate(
                        input_ids=ids, attention_mask=mask,
                        max_new_tokens=a.max_new_tokens, do_sample=False,
                        repetition_penalty=1.03, pad_token_id=tok.eos_token_id)
                answer = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            except Exception as exc:
                skipped += 1
                print(json.dumps({"event": "skip", "seen": seen, "error": str(exc)}, ensure_ascii=False), flush=True)
                continue
            if len(answer.split()) < 2:
                skipped += 1
                continue
            out.write(json.dumps({
                "module": "voz", "regime": "supervised_distillation_qwen_0.5b",
                "prompt": row["prompt"], "response": answer,
                "target_suffix": "<|im_end|><|end_of_text|>",
                "teacher": "Qwen/Qwen2.5-0.5B-Instruct",
            }, ensure_ascii=False) + "\n")
            written += 1
            print(json.dumps({"event": "progress", "seen": seen,
                              "written": written, "skipped": skipped}, ensure_ascii=False), flush=True)
    manifest = {"teacher": "Qwen/Qwen2.5-0.5B-Instruct",
                "teacher_parameters": 494032768, "source": str(a.data),
                "output": str(a.output), "requested": len(rows),
                "written": written, "skipped": skipped, "dtype": "float32",
                "max_input_tokens": a.max_input_tokens,
                "max_new_tokens": a.max_new_tokens, "sampling": "greedy"}
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

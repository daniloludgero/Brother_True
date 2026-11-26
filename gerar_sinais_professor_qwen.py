"""Gera sinais SFT supervisionados offline usando Qwen como professor."""

from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MESSAGE_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)


def messages_from_prompt(prompt: str):
    messages = [{"role": role, "content": content.strip()}
                for role, content in MESSAGE_RE.findall(prompt)]
    return [m for m in messages if m["content"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--max-input-tokens", type=int, default=384)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=20260824)
    a = p.parse_args()

    random.seed(a.seed)
    torch.set_num_threads(2)
    records = []
    with a.data.open(encoding="utf-8") as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if messages_from_prompt(item.get("prompt", "")):
                records.append(item)
    random.shuffle(records)
    records = records[:a.limit]

    tokenizer = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(a.teacher), local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with a.output.open("w", encoding="utf-8") as out:
        for index, item in enumerate(records, 1):
            messages = messages_from_prompt(item["prompt"])
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                input_ids = encoded["input_ids"][:, -a.max_input_tokens:]
                attention_mask = encoded["attention_mask"][:, -a.max_input_tokens:]
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=a.max_new_tokens, do_sample=False,
                        repetition_penalty=1.03,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                answer = tokenizer.decode(
                    generated[0, input_ids.shape[-1]:], skip_special_tokens=True
                ).strip()
            except Exception as exc:
                skipped += 1
                print(json.dumps({"event": "skip", "index": index,
                                  "error": str(exc)}, ensure_ascii=False), flush=True)
                continue
            if len(answer.split()) < 2:
                skipped += 1
                continue
            result = {
                "module": "voz",
                "regime": "supervised_distillation_qwen_0.5b",
                "prompt": item["prompt"],
                "response": answer,
                "target_suffix": "<|im_end|><|end_of_text|>",
                "teacher": "Qwen/Qwen2.5-0.5B-Instruct",
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            if index == 1 or index % 25 == 0 or index == len(records):
                print(json.dumps({"event": "progress", "seen": index,
                                  "written": written, "skipped": skipped},
                                 ensure_ascii=False), flush=True)

    manifest = {
        "teacher": "Qwen/Qwen2.5-0.5B-Instruct",
        "teacher_parameters": 494032768,
        "source": str(a.data),
        "output": str(a.output),
        "requested": len(records),
        "written": written,
        "skipped": skipped,
        "max_input_tokens": a.max_input_tokens,
        "max_new_tokens": a.max_new_tokens,
        "sampling": "greedy",
    }
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

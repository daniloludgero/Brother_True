"""Gera SFT curado: contexto começa no usuário e resposta contém terminadores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MESSAGE_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)


def extract(text: str, max_context: int):
    messages = [(r, c.strip()) for r, c in MESSAGE_RE.findall(text)]
    for i, (role, response) in enumerate(messages):
        if role != "assistant" or len(response.split()) < 2 or i == 0:
            continue
        if messages[i - 1][0] != "user":
            continue
        context = messages[max(0, i - max_context):i]
        while context and context[0][0] != "user":
            context = context[1:]
        if not context or context[-1][0] != "user":
            continue
        prompt = "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in context)
        prompt += "<|im_start|>assistant\n"
        yield {"module": "voz", "regime": "sft_chat_curado",
               "prompt": prompt, "response": response,
               "target_suffix": "<|im_end|><|end_of_text|>"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--max-context", type=int, default=4)
    a = p.parse_args()
    written = 0; conversations = 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.source.open(encoding="utf-8") as source, a.output.open("w", encoding="utf-8") as out:
        for line in source:
            if written >= a.limit: break
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            conversations += 1
            for example in extract(item.get("text", ""), a.max_context):
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                if written >= a.limit: break
    print(json.dumps({"conversations": conversations, "examples": written,
                      "output": str(a.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

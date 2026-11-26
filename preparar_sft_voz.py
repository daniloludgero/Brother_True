"""Converte diálogos humanos em pares contexto -> resposta para SFT."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MESSAGE_RE = re.compile(
    r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S
)


def examples_from_text(text: str, max_context: int):
    messages = [(role, content.strip()) for role, content in MESSAGE_RE.findall(text)]
    for index, (role, response) in enumerate(messages):
        if role != "assistant" or not response:
            continue
        previous = messages[max(0, index - max_context):index]
        if not previous or previous[-1][0] != "user":
            continue
        prompt_parts = []
        for prev_role, content in previous:
            prompt_parts.append(f"<|im_start|>{prev_role}\n{content}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        yield {
            "module": "voz",
            "regime": "sft_conversa_humana",
            "prompt": "".join(prompt_parts),
            "response": response,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=150000)
    p.add_argument("--max-context", type=int, default=4)
    a = p.parse_args()

    written = 0
    conversations = 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.source.open("r", encoding="utf-8") as source, a.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if written >= a.limit:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversations += 1
            for example in examples_from_text(item.get("text", ""), a.max_context):
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                if written >= a.limit:
                    break
    print(json.dumps({"conversations": conversations, "examples": written,
                      "output": str(a.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

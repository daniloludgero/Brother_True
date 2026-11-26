"""Prepara pares para a Voz sem instruções de raciocínio no prompt.

O conjunto bruto contém prompts com blocos <system> que pedem <think>.
Essas instruções pertencem à classe Pensamento, não à resposta final da Voz.
Aqui preservamos o contexto conversacional, removemos system/think e
garantimos que cada prompt termine em <assistant>.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
SYSTEM_RE = re.compile(r"<system>.*?(?=<user>|<assistant>|$)", re.I | re.S)
TAG_RE = re.compile(r"\n{3,}")


def clean_prompt(prompt: str) -> str:
    text = SYSTEM_RE.sub("", prompt)
    text = THINK_RE.sub("", text)
    text = re.sub(r"</?think>", "", text, flags=re.I)
    text = text.replace("\r\n", "\n").strip()
    if "<user>" in text:
        text = text[text.find("<user>"):]
    text = TAG_RE.sub("\n\n", text)
    if not text.endswith("<assistant>"):
        text = text.rstrip() + "\n<assistant>"
    return text + "\n"


def clean_response(response: str) -> str:
    text = THINK_RE.sub("", response)
    text = re.sub(r"</?think>", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"input": 0, "written": 0, "discarded": 0}
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            counts["input"] += 1
            item = json.loads(line)
            prompt = clean_prompt(item.get("prompt", ""))
            response = clean_response(item.get("response", ""))
            if not prompt or not response or "<think>" in prompt.lower():
                counts["discarded"] += 1
                continue
            dst.write(json.dumps({
                "source": item.get("source"),
                "row": item.get("row"),
                "prompt": prompt,
                "response": response,
            }, ensure_ascii=False) + "\n")
            counts["written"] += 1
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

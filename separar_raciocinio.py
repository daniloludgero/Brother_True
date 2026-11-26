"""Separa resposta final e trace de raciocínio para objetivos distintos."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.I | re.S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--reasoning", type=Path, required=True)
    args = p.parse_args()
    args.responses.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning.parent.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "with_reasoning": 0}
    with args.input.open(encoding="utf-8") as src, args.responses.open("w", encoding="utf-8") as out, args.reasoning.open("w", encoding="utf-8") as thought:
        for line in src:
            item = json.loads(line); response = item.get("response", "")
            match = THINK_RE.search(response)
            reasoning = match.group(1).strip() if match else ""
            final = THINK_RE.sub("", response).strip()
            final = re.sub(r"^\s*\n+", "", final)
            if not final: continue
            base = {"source": item.get("source"), "row": item.get("row"), "prompt": item.get("prompt", "")}
            out.write(json.dumps({**base, "response": final}, ensure_ascii=False) + "\n")
            counts["total"] += 1
            if reasoning:
                thought.write(json.dumps({**base, "reasoning": reasoning, "final": final}, ensure_ascii=False) + "\n")
                counts["with_reasoning"] += 1
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

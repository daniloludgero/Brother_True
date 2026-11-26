"""Extrai pares contexto-resposta dos datasets conversacionais."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def parse_messages(value: str):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []


def extract(messages):
    if not isinstance(messages, list):
        return None
    assistant_positions = [i for i, m in enumerate(messages)
                           if isinstance(m, dict) and m.get("role") == "assistant"
                           and m.get("content")]
    if not assistant_positions:
        return None
    end = assistant_positions[-1]
    response = str(messages[end].get("content", "")).strip()
    context = []
    for message in messages[:end]:
        if not isinstance(message, dict) or not message.get("content"):
            continue
        role = str(message.get("role", "user"))
        content = str(message["content"]).strip()
        context.append(f"<{role}>\n{content}")
    if not context or not response:
        return None
    return "\n".join(context) + "\n<assistant>\n", response


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("dados_preparados/dialogos.jsonl"))
    args = p.parse_args()
    csv.field_size_limit(2**31 - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as source:
                for row_number, row in enumerate(csv.DictReader(source), start=1):
                    pair = extract(parse_messages(row.get("messages", "")))
                    if pair is None:
                        continue
                    prompt, response = pair
                    record = {"source": path.name, "row": row_number,
                              "prompt": prompt, "response": response}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[path.name] = counts.get(path.name, 0) + 1
                    total += 1
    print(json.dumps({"total": total, "by_file": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

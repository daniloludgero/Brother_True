"""Extrai diálogos com parser de campos quoted, sem executar conteúdo."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELD_RE = re.compile(r"['\"]role['\"]\s*:\s*['\"]([^'\"]+)['\"]|['\"]content['\"]\s*:\s*(['\"])", re.S)


def fields(value: str):
    matches = list(FIELD_RE.finditer(value))
    result = []
    current_role = None
    for match in matches:
        if match.group(1):
            current_role = match.group(1)
            continue
        quote = match.group(2)
        start = match.end()
        i = start
        escaped = False
        while i < len(value):
            char = value[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                tail = value[i + 1:i + 4]
                if tail.startswith(",") or tail.startswith("}") or tail.startswith("\n"):
                    break
            i += 1
        if current_role and i > start:
            raw = value[start:i]
            try:
                content = bytes(raw, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                content = raw
            result.append((current_role, content))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("dados_preparados/dialogos.jsonl"))
    args = p.parse_args()
    csv.field_size_limit(2**31 - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, total = {}, 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as source:
                for row_number, row in enumerate(csv.DictReader(source), start=1):
                    messages = fields(row.get("messages", ""))
                    assistant = [i for i, (role, content) in enumerate(messages)
                                 if role == "assistant" and content.strip()]
                    if not assistant:
                        continue
                    end = assistant[-1]
                    response = messages[end][1].strip()
                    context = [f"<{role}>\n{content.strip()}" for role, content in messages[:end]
                               if content.strip()]
                    if not context or not response:
                        continue
                    record = {"source": path.name, "row": row_number,
                              "prompt": "\n".join(context) + "\n<assistant>\n",
                              "response": response}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[path.name] = counts.get(path.name, 0) + 1
                    total += 1
    print(json.dumps({"total": total, "by_file": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

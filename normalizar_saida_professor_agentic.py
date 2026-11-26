"""Normaliza a saída do professor para o contrato final_answer/tool_calls."""

from __future__ import annotations

import json
import re
from typing import Any


def normalize(text: str) -> dict[str, Any]:
    raw = text.strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            obj, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("final_answer" in obj or "tool_calls" in obj):
            calls = obj.get("tool_calls", []) or []
            normalized = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name", call.get("tool"))
                arguments = call.get("arguments", {})
                if name:
                    normalized.append({"name": name, "arguments": arguments})
            return {"final_answer": str(obj.get("final_answer", "")).strip(),
                    "tool_calls": normalized}
    calls = []
    pattern = re.compile(r"\{\s*\"(?:tool|name)\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*(\{.*?\})\s*\}", re.S)
    for match in pattern.finditer(raw):
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": match.group(1), "arguments": args})
    if calls:
        prefix = raw[:raw.find("{")].strip().split("\n", 1)[0]
        return {"final_answer": prefix, "tool_calls": calls}
    return {"final_answer": raw, "tool_calls": []}


if __name__ == "__main__":
    import sys
    print(json.dumps(normalize(" ".join(sys.argv[1:])), ensure_ascii=False))

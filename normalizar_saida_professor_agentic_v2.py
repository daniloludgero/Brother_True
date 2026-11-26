"""Normalizador robusto para saídas agentic do professor Qwen."""

from __future__ import annotations

import json
import re
from typing import Any


def _name_from_text(text: str) -> str | None:
    match = re.search(r"[`']([A-Za-z_][A-Za-z0-9_]*)[`']", text)
    return match.group(1) if match else None


def normalize(text: str) -> dict[str, Any]:
    raw = text.strip()
    decoder = json.JSONDecoder()
    parsed = []
    for match in re.finditer(r"\{", raw):
        try:
            obj, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    for obj in parsed:
        final = str(obj.get("final_answer", "")).strip()
        calls = obj.get("tool_calls")
        if isinstance(calls, list):
            normalized = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name", call.get("tool"))
                if name:
                    normalized.append({"name": name,
                                       "arguments": call.get("arguments", {})})
            return {"final_answer": final, "tool_calls": normalized}
        args = obj.get("arguments")
        if isinstance(args, list):
            name = obj.get("name") or obj.get("tool") or _name_from_text(final)
            if name:
                return {"final_answer": final,
                        "tool_calls": [{"name": name, "arguments": item}
                                       for item in args if isinstance(item, dict)]}
    pattern = re.compile(
        r"\{\s*\"(?:tool|name)\"\s*:\s*\"([^\"]+)\"\s*,\s*"
        r"\"arguments\"\s*:\s*(\{.*?\})\s*\}", re.S)
    calls = []
    for match in pattern.finditer(raw):
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": match.group(1), "arguments": args})
    if calls:
        return {"final_answer": raw[:raw.find("{")].strip(), "tool_calls": calls}
    return {"final_answer": raw, "tool_calls": []}


if __name__ == "__main__":
    import sys
    print(json.dumps(normalize(" ".join(sys.argv[1:])), ensure_ascii=False))

"""Filtra sinais roteados antes da destilação da Voz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def valid_complex(response: str) -> bool:
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        return False
    final = str(obj.get("final_answer", ""))
    low = final.lower()
    if not 2 <= len(final.split()) <= 160:
        return False
    if any(marker in low for marker in ("toolingtooling", "<commit_msg>", "<commit_after>",
                                        "{\"final_answer\":{\"final_answer\"")):
        return False
    if low.count("exploit") > 4 or low.count("tooling") > 4:
        return False
    calls = obj.get("tool_calls", [])
    if not isinstance(calls, list):
        return False
    return all(isinstance(c, dict) and c.get("name") and isinstance(c.get("arguments", {}), dict)
               for c in calls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    kept = dropped = 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.input.open(encoding="utf-8") as source, a.output.open("w", encoding="utf-8") as out:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            ok = True
            if row.get("complexity") == "complex":
                ok = valid_complex(row.get("response", ""))
            else:
                text = str(row.get("response", ""))
                ok = 2 <= len(text.split()) <= 160 and "as an ai language model" not in text.lower()
            if ok:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    result = {"input": str(a.input), "output": str(a.output), "kept": kept, "dropped": dropped}
    a.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

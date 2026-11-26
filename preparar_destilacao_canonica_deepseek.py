"""Separa think privado e resposta agentic canônica do Parquet DeepSeek."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_response(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_calls(value):
    calls = []
    if not isinstance(value, list):
        return calls
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = item.get("name") or item.get("tool") or function.get("name")
        arguments = item.get("arguments", function.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if name:
            calls.append({"name": str(name), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def canonical(raw):
    obj = parse_response(raw)
    message = obj.get("raw_message") if isinstance(obj.get("raw_message"), dict) else {}
    final = obj.get("final_answer", "")
    if not final:
        final = message.get("content", "")
    calls = normalize_calls(obj.get("tool_calls"))
    if not calls:
        calls = normalize_calls(message.get("tool_calls"))
    think = obj.get("reasoning") or obj.get("think") or message.get("reasoning") or ""
    final = str(final or "").strip()
    think = str(think or "").strip()
    if not final and not calls:
        return None
    return {"final_answer": final, "tool_calls": calls}, think


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    rows = []
    stats = {"total": 0, "accepted": 0, "rejected": 0, "with_final_answer": 0,
             "with_tool_calls": 0, "with_think": 0, "with_both": 0}
    parquet = pq.ParquetFile(a.input)
    for batch in parquet.iter_batches(batch_size=512,
                                      columns=["prompt", "response", "kind", "difficulty", "bucket"]):
        for source in batch.to_pylist():
            stats["total"] += 1
            normalized = canonical(source.get("response"))
            if normalized is None or not str(source.get("prompt", "")).strip():
                stats["rejected"] += 1
                continue
            target, think = normalized
            has_final = bool(target["final_answer"])
            has_calls = bool(target["tool_calls"])
            stats["accepted"] += 1
            stats["with_final_answer"] += int(has_final)
            stats["with_tool_calls"] += int(has_calls)
            stats["with_think"] += int(bool(think))
            stats["with_both"] += int(has_final and has_calls)
            rows.append({"prompt": source["prompt"],
                         "response": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                         "think": think,
                         "kind": source.get("kind") or "",
                         "difficulty": source.get("difficulty") or "",
                         "bucket": source.get("bucket") or ""})
    table = pa.Table.from_pylist(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, a.output, compression="zstd")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"input": str(a.input), "output": str(a.output), **stats},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

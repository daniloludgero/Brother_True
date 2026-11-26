"""Extrai exemplos canônicos completos: final_answer + tool_calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    rows, total = [], 0
    parquet = pq.ParquetFile(a.input)
    for batch in parquet.iter_batches(batch_size=512, columns=["prompt", "response", "think", "kind", "difficulty", "bucket"]):
        for row in batch.to_pylist():
            total += 1
            try:
                target = json.loads(row["response"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(target, dict) or not str(target.get("final_answer", "")).strip():
                continue
            calls = target.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                continue
            rows.append(row)
    table = pa.Table.from_pylist(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, a.output, compression="zstd")
    report = {"input": str(a.input), "output": str(a.output), "total": total,
              "schema_complete": len(rows), "requires_final_answer": True,
              "requires_tool_calls": True}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

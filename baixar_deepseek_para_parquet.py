"""Baixa o dataset DeepSeek por streaming e grava uma cópia Parquet local."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


FIELDS = ["prompt", "response", "kind", "difficulty", "bucket", "generation_mode",
          "finish_reason", "secondary_capabilities", "tool_definitions", "source",
          "teacher", "usage", "metadata"]


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="trjxter/DeepSeek-V4-Flash-0731-Teacher-Distillation-40513x")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.repo, split="train", streaming=True)
    writer = None
    batch = []
    count = 0
    try:
        for row in ds:
            metadata = {k: row.get(k) for k in (
                "api_retry_count", "attempt", "generated_at_unix", "latency_seconds",
                "manifest_index", "quality_attempt", "rate_limit_hits", "seed_id")}
            item = {
                "prompt": stringify(row.get("prompt")),
                "response": stringify(row.get("response")),
                "kind": stringify(row.get("kind")),
                "difficulty": stringify(row.get("difficulty")),
                "bucket": stringify(row.get("bucket")),
                "generation_mode": stringify(row.get("generation_mode")),
                "finish_reason": stringify(row.get("finish_reason")),
                "secondary_capabilities": stringify(row.get("secondary_capabilities")),
                "tool_definitions": stringify(row.get("tool_definitions")),
                "source": stringify(row.get("source")),
                "teacher": stringify(row.get("teacher")),
                "usage": stringify(row.get("usage")),
                "metadata": stringify(metadata),
            }
            batch.append(item)
            count += 1
            if len(batch) >= a.batch_size:
                table = pa.Table.from_pylist(batch, schema=pa.schema([(f, pa.string()) for f in FIELDS]))
                if writer is None:
                    writer = pq.ParquetWriter(str(a.output), table.schema, compression="zstd")
                writer.write_table(table)
                batch.clear()
                if count % (a.batch_size * 4) == 0:
                    free = shutil.disk_usage(a.output.anchor or str(a.output)).free / 1024**3
                    print(json.dumps({"event": "progress", "rows": count,
                                      "free_gb": round(free, 2)}, ensure_ascii=False), flush=True)
        if batch:
            table = pa.Table.from_pylist(batch, schema=pa.schema([(f, pa.string()) for f in FIELDS]))
            if writer is None:
                writer = pq.ParquetWriter(str(a.output), table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    result = {"repo": a.repo, "rows": count, "output": str(a.output),
              "format": "parquet", "compression": "zstd", "fields": FIELDS}
    a.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Prepara sandbox isolada: treino aprovado e fila de revisão DeepSeek."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


TEACHER = "deepseek-ai/DeepSeek-V4-Flash-0731"


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def approved_from_parquet(path: Path, encoder_limit: int, decoder_limit: int):
    encoder, decoder = [], []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["prompt", "response", "kind", "teacher"]):
        for row in batch.to_pylist():
            if len(decoder) >= decoder_limit and len(encoder) >= encoder_limit:
                return encoder, decoder
            try:
                target = json.loads(row["response"])
            except (TypeError, json.JSONDecodeError):
                continue
            final = str(target.get("final_answer", "")).strip() if isinstance(target, dict) else ""
            if len(decoder) < decoder_limit:
                decoder.append({"prompt": row["prompt"], "response": row["response"],
                                "kind": row.get("kind") or "", "teacher": TEACHER,
                                "teacher_approved": True, "source": str(path)})
            if final and len(encoder) < encoder_limit:
                encoder.append({"text_a": row["prompt"], "text_b": final, "label": 1.0,
                                "kind": row.get("kind") or "", "teacher": TEACHER,
                                "teacher_approved": True, "source": str(path)})
    return encoder, decoder


def candidate_rows(path: Path, role: str, limit: int):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(rows) >= limit:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt", "")
            response = item.get("response", item.get("final", ""))
            text = item.get("text", "")
            if not prompt and not text and not response:
                continue
            rows.append({"candidate_id": f"{path.name}:{index}", "role": role,
                         "prompt": prompt, "text": text, "response": response,
                         "source": str(path), "teacher_approved": False,
                         "review_required": TEACHER})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--approved-parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-source", type=int, default=128)
    p.add_argument("--encoder-limit", type=int, default=2048)
    p.add_argument("--decoder-limit", type=int, default=4096)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "checkpoints").mkdir(exist_ok=True)
    encoder, decoder = approved_from_parquet(a.approved_parquet, a.encoder_limit, a.decoder_limit)
    write_jsonl(a.out / "teacher_approved_encoder.jsonl", encoder)
    write_jsonl(a.out / "teacher_approved_decoder.jsonl", decoder)

    sources = [
        (a.root / "dados_modulos_novos" / "voz_natural.jsonl", "natural_language"),
        (a.root / "dados_modulos_novos" / "codigo_secundario.jsonl", "code"),
        (a.root / "dados_modulos_novos" / "pensamento_think.jsonl", "private_think"),
        (a.root / "dados_preparados" / "dialogos_robustos.jsonl", "dialogue"),
        (a.root / "dados_preparados" / "raciocinio_pensamento.jsonl", "reasoning"),
        (a.root / "dados_externos" / "tokenizers_english_sample.jsonl", "english"),
        (a.root / "dados_externos" / "tokenizers_code_sample.jsonl", "code"),
    ]
    queue, source_stats = [], []
    for path, role in sources:
        if not path.exists():
            continue
        selected = candidate_rows(path, role, a.per_source)
        queue.extend(selected)
        source_stats.append({"source": str(path), "role": role, "selected": len(selected),
                             "bytes": path.stat().st_size})
    write_jsonl(a.out / "deepseek_review_queue.jsonl", queue)
    manifest = {
        "experiment": "isolated_deepseek_encoder_decoder_2026-08-27",
        "teacher": TEACHER,
        "approved_training": {"encoder_rows": len(encoder), "decoder_rows": len(decoder),
                               "source": str(a.approved_parquet), "teacher_approved": True},
        "review_queue": {"rows": len(queue), "teacher_approved": False,
                          "policy": "never train until DeepSeek validates each candidate"},
        "sources": source_stats,
        "checkpoints_dir": str(a.out / "checkpoints"),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

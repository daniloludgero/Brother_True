"""Normaliza os CSVs fornecidos para JSONL e corpus de tokenizer.

Processamento streaming: os arquivos não são carregados inteiros na memória.
Campos de mensagens são analisados com ast.literal_eval, nunca com eval.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def _messages_to_text(value: str) -> str:
    try:
        messages = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError):
        return value
    if not isinstance(messages, list):
        return str(messages)
    parts = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "unknown"))
            content = message.get("content", "")
            if content is None:
                continue
            parts.append(f"<{role}> {content}")
    return "\n".join(parts)


def _record(path: Path, row: dict[str, str]) -> tuple[str, str] | None:
    name = path.name
    if name.startswith("ultrafineweb"):
        return "web", row.get("content", "")
    if name.startswith("qwen_"):
        return "math_distillation", _messages_to_text(row.get("messages", ""))
    if name.startswith("fable_"):
        return "agent_dialogue", _messages_to_text(row.get("messages", ""))
    if name.startswith("deepseek_"):
        return "agent_trace", _messages_to_text(row.get("messages", ""))
    if name == "terminal_bench_full.csv":
        text = "\n".join(row.get(k, "") for k in ("name", "category", "difficulty", "keywords", "description", "instruction"))
        return "terminal_task", text
    if name == "terminal_bench_summary.csv":
        return "benchmark_summary", f"Categoria: {row.get('Categoria', '')}. Quantidade: {row.get('Quantidade', '')}."
    return None


def convert(inputs: list[Path], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "dataset_normalizado.jsonl"
    corpus_path = output_dir / "corpus_tokenizer.txt"
    counts: dict[str, int] = {}
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, corpus_path.open("w", encoding="utf-8", newline="\n") as corpus:
        for path in inputs:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
                reader = csv.DictReader(source)
                for row_number, row in enumerate(reader, start=1):
                    item = _record(path, row)
                    if item is None:
                        continue
                    source_name, text = item
                    text = text.replace("\x00", " ").strip()
                    if not text:
                        continue
                    record = {"source": source_name, "file": path.name, "row": row_number, "text": text}
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    corpus.write(text.replace("\r", " ").replace("\n", " ") + "\n")
                    counts[source_name] = counts.get(source_name, 0) + 1
    (output_dir / "manifest.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dados_preparados"))
    args = parser.parse_args()
    print(json.dumps(convert(args.inputs, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

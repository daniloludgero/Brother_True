"""Organiza a nova linha de dados sem misturar treino e avaliação."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_text(path: Path, module: str, regime: str):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = item.get("text", "").strip()
            if text:
                rows.append({"module": module, "regime": regime,
                             "text": text, "source": item.get("source"),
                             "config": item.get("config")})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--externos", type=Path, required=True)
    parser.add_argument("--pensamento", type=Path, required=True)
    parser.add_argument("--farmaco-ddi", type=Path, required=True)
    parser.add_argument("--farmaco-bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    voz = raw_text(args.externos / "human_chats_sample.jsonl", "voz", "natural_dialogue")
    voz += raw_text(args.externos / "tokenizers_english_sample.jsonl", "voz", "natural_language")
    codigo = raw_text(args.externos / "tokenizers_code_sample.jsonl", "codigo", "technical_code")
    write_jsonl(args.out / "voz_natural.jsonl", voz)
    write_jsonl(args.out / "codigo_secundario.jsonl", codigo)

    thought = []
    with args.pensamento.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("reasoning"):
                thought.append({"module":"pensamento", "regime":"reasoning_private",
                                "prompt":item.get("prompt",""),
                                "response":item["reasoning"],
                                "final":item.get("final",""),
                                "source":item.get("source"), "row":item.get("row")})
    write_jsonl(args.out / "pensamento_think.jsonl", thought)

    # Estes dois arquivos têm nome de teste/benchmark: ficam exclusivamente
    # fora do treino para evitar vazamento de avaliação.
    ddi = []
    with args.farmaco_ddi.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            ddi.append({"module":"farmacologia", "regime":"factual_eval",
                        "task":"drug_drug_interaction",
                        "prompt":item.get("Question", ""),
                        "reference":item.get("Drugs-drug interaction_reference", ""),
                        "model_answer":item.get("GPT_3.5_rep1", ""),
                        "drug":item.get("Name"), "drug_id":item.get("Parent_key")})
    write_jsonl(args.out / "farmacologia_ddi_eval.jsonl", ddi)

    bench = []
    with args.farmaco_bench.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            options = {key: item.get(key) for key in ("A", "B", "C", "D")}
            bench.append({"module":"farmacologia", "regime":"factual_eval",
                          "task":"multiple_choice",
                          "prompt":item.get("question", ""),
                          "options":options, "answer":item.get("answer"),
                          "subject":item.get("subject"),
                          "exam_level":item.get("exam_level")})
    write_jsonl(args.out / "farmacologia_bench_eval.jsonl", bench)

    manifest = {
        "training": {
            "voz_natural.jsonl": len(voz),
            "pensamento_think.jsonl": len(thought),
            "codigo_secundario.jsonl": len(codigo),
        },
        "evaluation_only": {
            "farmacologia_ddi_eval.jsonl": len(ddi),
            "farmacologia_bench_eval.jsonl": len(bench),
        },
        "excluded_from_new_voice": ["dialogos_voz_limpos.jsonl", "dataset_normalizado.jsonl"],
        "frozen_rag_policy": "old mixed data remains retrieval-only until reclassified",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

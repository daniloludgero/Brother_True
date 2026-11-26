"""Separa o corpus de Voz em namespaces explícitos para treino e GraphRAG."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_TOPIC = {
    "qwen_max_distillation_50k.csv": "matematica_raciocinio",
    "deepseek_agentic_canonical_full.csv": "codigo_agentes",
    "fable_premium_fixed_sample.csv": "dialogo_narrativo",
}


def topic_for(item: dict) -> str:
    source = item.get("source", "")
    if source in SOURCE_TOPIC:
        return SOURCE_TOPIC[source]
    text = f"{item.get('prompt', '')} {item.get('response', '')}".lower()
    if re.search(r"\b(código|code|python|java|javascript|terminal|shell|classe|function)\b", text):
        return "codigo_agentes"
    if re.search(r"\b(equação|paralelogramo|número|calcule|math|probabilidade|integral)\b", text):
        return "matematica_raciocinio"
    return "geral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    handles = {}
    counts = Counter()
    response_lengths = defaultdict(list)
    try:
        with args.input.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                topic = topic_for(item)
                if topic not in handles:
                    handles[topic] = (args.out / f"{topic}.jsonl").open("w", encoding="utf-8")
                item = {**item, "topic": topic}
                handles[topic].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[topic] += 1
                response_lengths[topic].append(len(item.get("response", "")))
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {"topics": {topic: {
        "records": counts[topic],
        "mean_response_chars": round(sum(response_lengths[topic]) / counts[topic], 2),
        "file": f"{topic}.jsonl",
    } for topic in sorted(counts)}, "reasoning_file": "raciocinio_pensamento.jsonl"}
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

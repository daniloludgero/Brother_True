"""Filtra a fila de treino por qualidade estrutural e relevância GraphRAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph_rag_hibrido import GraphRAGHibrido


def is_structurally_good(row: dict) -> bool:
    features = row.get("quality_features") or {}
    score = float(row.get("quality_score_proxy", 0.0))
    return (
        score >= 0.75
        and float(features.get("alignment_proxy", 0.0)) >= 0.65
        and float(features.get("final_supported_by_reasoning", 0.0)) >= 0.55
        and bool(features.get("reasoning_structure"))
        and bool(features.get("completeness"))
        and bool(features.get("length_ok"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.queue.open(encoding="utf-8") if line.strip()]
    structural = [row for row in rows if is_structurally_good(row)]
    retriever = GraphRAGHibrido(args.graph, args.semantic, args.minilm)
    scored = []
    try:
        for row in structural:
            hits = retriever.search(str(row.get("prompt", "")), top_k=3)
            top = hits[0] if hits else {}
            enriched = dict(row)
            enriched["graphrag_filter"] = True
            enriched["graphrag_top_score"] = float(top.get("score", 0.0))
            enriched["graphrag_top_semantic_score"] = float(top.get("semantic_score", 0.0))
            enriched["graphrag_top_record_index"] = top.get("record_index")
            enriched["graphrag_graph_expansion"] = bool(top.get("graph_expansion", False))
            scored.append(enriched)
    finally:
        retriever.close()
    scored.sort(key=lambda row: (row["graphrag_top_score"], row["quality_score_proxy"]), reverse=True)
    selected = scored[: min(args.keep, len(scored))]
    output_file = args.out / "filtered_dataset_graphrag.jsonl"
    with output_file.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "input_rows": len(rows),
        "structural_rows": len(structural),
        "scored_rows": len(scored),
        "selected_rows": len(selected),
        "keep": args.keep,
        "score_min": min((r["graphrag_top_score"] for r in scored), default=None),
        "score_max": max((r["graphrag_top_score"] for r in scored), default=None),
        "score_selected_min": min((r["graphrag_top_score"] for r in selected), default=None),
        "graph_expansion_selected": sum(bool(r["graphrag_graph_expansion"]) for r in selected),
        "output": str(output_file),
    }
    (args.out / "graphrag_filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    journal = Path(r"D:\brothertrue\training_memory_journal.jsonl")
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "graphrag_filter", **report}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Pareia pensamento_think/raciocinio_pensamento e cria fila para o DeepSeek."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]{3,}")
STOP = {"the", "and", "that", "this", "with", "from", "para", "como", "uma", "por", "que", "you", "are", "not"}
MARKERS = ("step", "verification", "therefore", "because", "calculate", "given", "passo", "verifica", "portanto")


def words(text: str) -> set[str]:
    return {x.lower() for x in WORD_RE.findall(text or "") if x.lower() not in STOP}


def load_map(path: Path, reasoning: bool) -> dict[tuple[str, str], dict]:
    data = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (str(row.get("source", "")), str(row.get("row", "")))
            if not key[0] or not key[1]:
                continue
            data[key] = row
    return data


def score_pair(think: dict, reasoning: dict) -> tuple[float, dict]:
    prompt = str(think.get("prompt", ""))
    private = str(think.get("response", "") or reasoning.get("reasoning", "")).strip()
    final = str(think.get("final", "") or reasoning.get("final", "")).strip()
    prompt_words = words(prompt)
    answer_words = words(private + " " + final)
    final_words = words(final)
    alignment = len(prompt_words & answer_words) / max(1, min(len(prompt_words), 80))
    final_support = len(final_words & words(private)) / max(1, len(final_words))
    structure = min(1.0, sum(marker in private.lower() for marker in MARKERS) / 3.0)
    completeness = float(bool(private)) * 0.55 + float(bool(final)) * 0.45
    length_ok = 1.0 if 80 <= len(private) <= 12000 else 0.5 if private else 0.0
    score = 0.30 * min(1.0, alignment * 2.0) + 0.25 * final_support + 0.20 * structure + 0.15 * completeness + 0.10 * length_ok
    features = {"alignment_proxy": round(min(1.0, alignment * 2.0), 4),
                "final_supported_by_reasoning": round(final_support, 4),
                "reasoning_structure": round(structure, 4),
                "completeness": round(completeness, 4), "length_ok": length_ok}
    return round(score, 4), features


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--think", type=Path, required=True)
    p.add_argument("--reasoning", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--limit", type=int, default=4096)
    a = p.parse_args()
    think_rows = load_map(a.think, False)
    reasoning_rows = load_map(a.reasoning, True)
    candidates, bands = [], Counter()
    for key, think in think_rows.items():
        reasoning = reasoning_rows.get(key)
        if not reasoning:
            continue
        value, features = score_pair(think, reasoning)
        band = "high" if value >= 0.70 else "medium" if value >= 0.45 else "low"
        bands[band] += 1
        candidates.append({"prompt": think.get("prompt", ""),
                           "think": think.get("response", ""),
                           "final": think.get("final", "") or reasoning.get("final", ""),
                           "source": think.get("source", ""), "row": think.get("row", ""),
                           "quality_score_proxy": value, "quality_features": features,
                           "teacher": "deepseek-ai/DeepSeek-V4-Flash-0731",
                           "teacher_approved": False, "review_required": True,
                           "review_policy": "DeepSeek must approve reasoning and final separately"})
    candidates.sort(key=lambda x: x["quality_score_proxy"], reverse=True)
    selected = candidates[:a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"think_rows": len(think_rows), "reasoning_rows": len(reasoning_rows),
              "paired_rows": len(candidates), "selected_rows": len(selected),
              "quality_bands_all_pairs": dict(bands), "output": str(a.out),
              "teacher": "deepseek-ai/DeepSeek-V4-Flash-0731",
              "teacher_approval_required": True,
              "training_status": "blocked_until_teacher_approval"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

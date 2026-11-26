"""Oráculo interno do módulo farmacológico; nunca expõe o benchmark ao usuário."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .farmacologia import CONSTANTES_NEUROQUIMICAS, estado_repouso, hill_occupancy


class AvaliadorFarmacologia:
    def __init__(self, ddi_path: str | Path, bench_path: str | Path):
        self.ddi = [json.loads(line) for line in Path(ddi_path).open(encoding="utf-8") if line.strip()]
        self.bench = [json.loads(line) for line in Path(bench_path).open(encoding="utf-8") if line.strip()]

    def score_multiple_choice(self, predictions: list[str]) -> dict:
        total = min(len(predictions), len(self.bench))
        correct = 0
        for prediction, item in zip(predictions[:total], self.bench[:total]):
            letter = re.search(r"\b([ABCD])\b", prediction.upper() if isinstance(prediction, str) else "")
            correct += int(letter is not None and letter.group(1) == str(item.get("answer", "")).upper())
        return {"total": total, "correct": correct, "accuracy": correct / max(1, total)}

    def invariant_report(self) -> dict:
        states = {name: estado_repouso(name) for name in CONSTANTES_NEUROQUIMICAS}
        midpoint_errors = {
            name: abs(hill_occupancy(value.kd_nm, value.kd_nm, value.n_hill) - 0.5)
            for name, value in CONSTANTES_NEUROQUIMICAS.items()
        }
        return {
            "ddi_reference_cases": len(self.ddi),
            "multiple_choice_cases": len(self.bench),
            "resting_state_in_range": all(0.0 <= value <= 1.0 for value in states.values()),
            "hill_midpoint_max_error": max(midpoint_errors.values()),
            "constants": states,
            "benchmark_visibility": "internal_only",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddi", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluator = AvaliadorFarmacologia(args.ddi, args.bench)
    report = evaluator.invariant_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

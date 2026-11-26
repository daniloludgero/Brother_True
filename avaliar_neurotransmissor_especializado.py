"""Validação do módulo farmacológico especializado, sem usar o benchmark como treino."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .farmacologia import CONSTANTES_NEUROQUIMICAS, estado_repouso, hill_occupancy
from .neuroquimica import NeuroTransmissor, ORDEM_NEUROTRANSMISSORES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddi", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    random.seed(20260823); torch.manual_seed(20260823)
    neuro = NeuroTransmissor()
    initial = neuro.estado_atual()
    midpoint_errors = {name: abs(hill_occupancy(c.kd_nm, c.kd_nm, c.n_hill) - 0.5)
                       for name, c in CONSTANTES_NEUROQUIMICAS.items()}
    violations = []
    finite_matrices = True
    for step in range(args.steps):
        e = random.uniform(0.0, 12.0)
        l = random.uniform(0.0, 1.0)
        q = random.uniform(0.0, 1.0)
        neuro.atualizar(e, l, q, collapsed=(step % 17 == 0))
        state = neuro.estado_atual()
        bad = {name: value for name, value in state.items()
               if not (0.0 <= value <= 1.0) or not torch.isfinite(torch.tensor(value))}
        if bad:
            violations.append({"step": step, "state": bad})
        matrices = neuro.get_modulation_matrices()
        finite_matrices = finite_matrices and all(
            matrix.shape == (2, 2) and bool(torch.isfinite(matrix).all())
            for matrix in matrices.values()
        )
    report = {
        "module": "NeuroTransmissor + farmacologia",
        "steps": args.steps,
        "initial_state": initial,
        "final_state": neuro.estado_atual(),
        "neurotransmitter_order": ORDEM_NEUROTRANSMISSORES,
        "state_range_violations": len(violations),
        "finite_modulation_matrices": finite_matrices,
        "hill_midpoint_max_error": max(midpoint_errors.values()),
        "ddi_reserved_cases": sum(1 for line in args.ddi.open(encoding="utf-8") if line.strip()),
        "benchmark_reserved_cases": sum(1 for line in args.bench.open(encoding="utf-8") if line.strip()),
        "benchmark_used_for_training": False,
        "benchmark_used_for_state_updates": False,
        "violations_preview": violations[:3],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["state_range_violations"] or not report["finite_modulation_matrices"] or report["hill_midpoint_max_error"] > 1e-9:
        raise SystemExit(1)


if __name__ == "__main__": main()

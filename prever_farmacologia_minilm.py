"""Baseline interno: seleciona alternativas farmacológicas com MiniLM congelado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .avaliador_farmacologia_interno import AvaliadorFarmacologia


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddi", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluator = AvaliadorFarmacologia(args.ddi, args.bench)
    model = SentenceTransformer(str(args.minilm), device="cpu", local_files_only=True)
    model.eval()
    letters = ["A", "B", "C", "D"]
    questions = [item["prompt"] for item in evaluator.bench]
    pairs = [f"{item['prompt']}\nOption {letter}: {item.get('options', {}).get(letter, '')}"
             for item in evaluator.bench for letter in letters]
    question_vec = model.encode(questions, batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
    pair_vec = model.encode(pairs, batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    scores = (pair_vec @ np.repeat(question_vec, 4, axis=0).T.diagonal()).reshape(-1, 4)
    predictions = [letters[int(row.argmax())] for row in scores]
    mcq = evaluator.score_multiple_choice(predictions)

    ddi_scores = []
    if evaluator.ddi:
        refs = [item.get("reference", "") for item in evaluator.ddi]
        answers = [item.get("model_answer", "") for item in evaluator.ddi]
        ref_vec = model.encode(refs, batch_size=16, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        ans_vec = model.encode(answers, batch_size=16, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        ddi_scores = (ref_vec * ans_vec).sum(axis=1).tolist()
    report = {
        "model": "all-MiniLM-L6-v2 frozen semantic baseline",
        "multiple_choice": mcq,
        "ddi": {
            "cases": len(ddi_scores),
            "reference_similarity_mean": float(np.mean(ddi_scores)) if ddi_scores else 0.0,
            "reference_similarity_min": float(np.min(ddi_scores)) if ddi_scores else 0.0,
        },
        "visibility": "internal_only",
        "warning": "This is a retrieval/semantic baseline, not a trained pharmacology predictor.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

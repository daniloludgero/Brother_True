"""Baseline interno corrigido: seleção semântica A-D com MiniLM congelado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .avaliador_farmacologia_interno import AvaliadorFarmacologia


def main():
    p=argparse.ArgumentParser(); p.add_argument("--ddi",type=Path,required=True); p.add_argument("--bench",type=Path,required=True); p.add_argument("--minilm",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2); evaluator=AvaliadorFarmacologia(a.ddi,a.bench); model=SentenceTransformer(str(a.minilm),device="cpu",local_files_only=True); model.eval(); letters=["A","B","C","D"]
    questions=[item["prompt"] for item in evaluator.bench]
    pairs=[f"{item['prompt']}\nOption {letter}: {item.get('options',{}).get(letter,'')}" for item in evaluator.bench for letter in letters]
    q=model.encode(questions,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    pair=model.encode(pairs,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    scores=(pair*np.repeat(q,4,axis=0)).sum(axis=1).reshape(-1,4); predictions=[letters[int(row.argmax())] for row in scores]; mcq=evaluator.score_multiple_choice(predictions)
    refs=[item.get("reference","") for item in evaluator.ddi]; answers=[item.get("model_answer","") for item in evaluator.ddi]
    ref_vec=model.encode(refs,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); ans_vec=model.encode(answers,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); ddi=(ref_vec*ans_vec).sum(axis=1)
    report={"model":"all-MiniLM-L6-v2 frozen semantic baseline","multiple_choice":mcq,"ddi":{"cases":len(ddi),"reference_similarity_mean":float(np.mean(ddi)),"reference_similarity_min":float(np.min(ddi))},"visibility":"internal_only","warning":"Baseline semântico; ainda não é um preditor farmacológico treinado."}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()

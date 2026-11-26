"""Avalia o checkpoint com embeddings e lm_head parcialmente ajustados."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .checkpoint_manager_safe import load_checkpoint
from .treinar_distilacao_dupla_voz import expanded_base
from .avaliar_distilacao_dupla_voz import generate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--base',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2); tok=Tokenizer.from_file(str(a.tokenizer)); model=expanded_base(a.base,16384,tok.get_vocab_size(),tok,128); payload=load_checkpoint(a.checkpoint,model=model,map_location='cpu'); model.eval(); rows=[json.loads(x) for x in a.data.read_text(encoding='utf-8').splitlines() if x.strip()]; examples=[{'complexity':r['complexity'],'teacher':r['teacher'],'generated':generate(model,tok,r['prompt'])} for r in rows[:2]+rows[-2:]]; result={'checkpoint_step':payload.get('step',150),'metrics':payload.get('metrics',{}),'examples':examples,'dataset_rows':len(rows)}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()

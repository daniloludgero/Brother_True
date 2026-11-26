"""Retoma o treino isolado do Pensamento textual."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .pensamento_linguistico import PensamentoLinguistico
from .resource_governor import ResourceGovernor, ResourcePolicy
from .treino_pensamento_textual import evaluate, example, offsets


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--base",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--steps",type=int,default=1500); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--max-len",type=int,default=128); a=p.parse_args()
    random.seed(20260823); torch.manual_seed(20260823); torch.set_num_threads(min(2,torch.get_num_threads()))
    tok=Tokenizer.from_file(str(a.tokenizer)); bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2; positions=offsets(a.data); random.shuffle(positions); val=positions[:max(1,len(positions)//10)]; train=positions[len(val):]
    model=PensamentoLinguistico(vocab_size=tok.get_vocab_size(),d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=a.max_len); opt=torch.optim.AdamW(model.parameters(),lr=a.lr); a.out.mkdir(parents=True,exist_ok=True); latest=a.out/"latest.pt"; best=a.out/"best.pt"; source=latest if latest.exists() else a.base; state=load_checkpoint(source,model=model,optimizer=opt,map_location="cpu"); step=int(state.get("step",0)); best_val=float(state.get("metrics",{}).get("val_loss",float("inf")))
    for group in opt.param_groups: group["lr"]=a.lr
    gov=ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,ram_limit_percent=90)); rng=random.Random(20260823+step)
    while step<a.steps:
        item=None
        while item is None: item=example(a.data,rng.choice(train),tok,a.max_len,bos,eos)
        x,y=item; model.train(); opt.zero_grad(set_to_none=True); logits=model(x.unsqueeze(0),torch.zeros(1,11)); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        if step%25==0 or step==a.steps:
            val_loss=evaluate(model,a.data,val,tok,a.max_len,bos,eos); metrics={"train_loss":float(loss.detach()),"val_loss":val_loss,"learning_rate":a.lr,"dataset":"think_only","resumed_from":str(source)}; save_checkpoint(latest,model=model,optimizer=opt,step=step,metrics=metrics)
            if val_loss<best_val: best_val=val_loss; save_checkpoint(best,model=model,optimizer=opt,step=step,metrics=metrics)
            print(json.dumps({"step":step,**metrics,"best_val":best_val,**gov.throttle()},ensure_ascii=False),flush=True)
    print(f"[OK] Pensamento retomado até o passo {step}; melhor validação {best_val:.5f}",flush=True)


if __name__ == "__main__": main()

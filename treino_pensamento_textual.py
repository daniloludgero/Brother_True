"""Treino inicial do submódulo textual Pensamento nos traces privados."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .checkpoint_manager_safe import save_checkpoint
from .pensamento_linguistico import PensamentoLinguistico
from .resource_governor import ResourceGovernor, ResourcePolicy


def offsets(path):
    out=[]
    with path.open("rb") as handle:
        while True:
            pos=handle.tell(); line=handle.readline()
            if not line: break
            if line.strip(): out.append(pos)
    return out


def example(path, pos, tok, max_len, bos, eos):
    with path.open("rb") as handle:
        handle.seek(pos); item=json.loads(handle.readline().decode("utf-8"))
    text=(item.get("prompt","")+"\n<think>\n"+item.get("response","")).strip()
    ids=tok.encode(text,add_special_tokens=False).ids
    if len(ids)<2: return None
    seq=[bos]+ids+[eos]
    if len(seq)>max_len:
        start=random.randint(0,len(seq)-max_len); seq=seq[start:start+max_len]; seq[0]=bos
    return torch.tensor(seq[:-1]),torch.tensor(seq[1:])


@torch.no_grad()
def evaluate(model,path,positions,tok,max_len,bos,eos):
    model.eval(); vals=[]
    for pos in positions[:24]:
        item=example(path,pos,tok,max_len,bos,eos)
        if item is None: continue
        x,y=item; logits=model(x.unsqueeze(0),torch.zeros(1,11))
        vals.append(float(F.cross_entropy(logits.reshape(-1,logits.size(-1)),y)))
    return sum(vals)/max(1,len(vals))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--steps",type=int,default=500); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--max-len",type=int,default=128); args=p.parse_args()
    random.seed(20260823); torch.manual_seed(20260823); torch.set_num_threads(min(2,torch.get_num_threads()))
    tok=Tokenizer.from_file(str(args.tokenizer)); bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2
    positions=offsets(args.data); random.shuffle(positions); val=positions[:max(1,len(positions)//10)]; train=positions[len(val):]
    model=PensamentoLinguistico(vocab_size=tok.get_vocab_size(),d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=args.max_len)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr); args.out.mkdir(parents=True,exist_ok=True); best=float("inf"); gov=ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,ram_limit_percent=90))
    for step in range(1,args.steps+1):
        item=None
        while item is None: item=example(args.data,random.choice(train),tok,args.max_len,bos,eos)
        x,y=item; model.train(); opt.zero_grad(set_to_none=True); logits=model(x.unsqueeze(0),torch.zeros(1,11)); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%25==0 or step==args.steps:
            val_loss=evaluate(model,args.data,val,tok,args.max_len,bos,eos); metrics={"train_loss":float(loss.detach()),"val_loss":val_loss,"dataset":"think_only","learning_rate":args.lr}; save_checkpoint(args.out/"latest.pt",model=model,optimizer=opt,step=step,metrics=metrics)
            if val_loss<best: best=val_loss; save_checkpoint(args.out/"best.pt",model=model,optimizer=opt,step=step,metrics=metrics)
            print(json.dumps({"step":step,**metrics,"best_val":best,**gov.throttle()},ensure_ascii=False),flush=True)
    print(f"[OK] Pensamento textual concluído; melhor validação {best:.5f}",flush=True)


if __name__ == "__main__": main()

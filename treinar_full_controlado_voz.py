"""Treino completo controlado da Voz 50M com dados de dois professores."""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from .checkpoint_manager_safe import save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .treinar_distilacao_dupla_voz import expanded_base, read_example, evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--base',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--steps',type=int,default=100); p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--max-len',type=int,default=128); p.add_argument('--save-every',type=int,default=25); a=p.parse_args()
    random.seed(20260825); torch.manual_seed(20260825); torch.set_num_threads(2)
    tok=Tokenizer.from_file(str(a.tokenizer)); positions=__import__('brothertrue.treino_voz_texto_limpo',fromlist=['offsets']).offsets(a.data); order=list(range(len(positions))); random.shuffle(order); val=order[:max(1,len(order)//5)]; train=order[len(val):]
    pad=tok.token_to_id('<pad>') or 0; bos=tok.token_to_id('<bos>') or 1; eos=tok.token_to_id('<eos>') or 2
    model=expanded_base(a.base,16384,tok.get_vocab_size(),tok,a.max_len); total=sum(x.numel() for x in model.parameters()); params=list(model.parameters()); trainable=sum(x.numel() for x in params); opt=torch.optim.AdamW(params,lr=a.lr); a.out.mkdir(parents=True,exist_ok=True); gov=ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,ram_limit_percent=90)); best=float('inf'); rng=random.Random(20260825)
    print(json.dumps({'event':'initialized','examples':len(positions),'total_parameters':total,'trainable_parameters':trainable,'regime':'full_controlled'},ensure_ascii=False),flush=True)
    for step in range(1,a.steps+1):
        x,y=read_example(a.data,positions[train[rng.randrange(len(train))]],tok,a.max_len,bos,eos); model.train(); opt.zero_grad(set_to_none=True); z=model(x.unsqueeze(0),torch.zeros(1,11)); loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y,ignore_index=-100); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if step%a.save_every==0 or step==a.steps:
            vl=evaluate(model,a.data,[positions[i] for i in val],tok,a.max_len,bos,eos); m={'train_loss':float(loss.detach()),'val_loss':vl,'step':step,'total_parameters':total,'trainable_parameters':trainable,'dataset':'two_teacher_qualified_full_controlled'}; save_checkpoint(a.out/'latest.pt',model=model,optimizer=opt,step=step,metrics=m)
            if vl<best: best=vl; save_checkpoint(a.out/'best.pt',model=model,optimizer=opt,step=step,metrics=m)
            print(json.dumps({'event':'checkpoint',**m,'best_val':best,**gov.throttle()},ensure_ascii=False),flush=True)
    print(json.dumps({'event':'complete','best_val':best,'out':str(a.out)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()

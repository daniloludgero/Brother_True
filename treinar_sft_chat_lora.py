"""SFT LoRA curado com delimitadores de conversa no alvo."""

from __future__ import annotations

import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from . import treinar_lora_voz
from .checkpoint_manager_safe import load_checkpoint, save_checkpoint
from .resource_governor import ResourceGovernor, ResourcePolicy
from .treino_voz_texto_limpo import offsets
from .voz import Voz

treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


def read_example(path, position, tok, max_len, bos, eos):
    with path.open("rb") as h:
        h.seek(position); item=json.loads(h.readline().decode("utf-8"))
    prompt=tok.encode(item["prompt"],add_special_tokens=False).ids
    target=tok.encode(item["response"]+item.get("target_suffix","<|im_end|><|end_of_text|>"),add_special_tokens=False).ids+[eos]
    target=target[:max_len-2]; prompt=prompt[-max(1,max_len-1-len(target)):]
    seq=[bos]+prompt+target; x=torch.tensor(seq[:-1]); y=torch.tensor(seq[1:])
    y[:max(0,len(prompt))]=-100
    return x,y


@torch.no_grad()
def evaluate(model,path,positions,tok,max_len,bos,eos):
    model.eval(); vals=[]
    for pos in positions[:32]:
        x,y=read_example(path,pos,tok,max_len,bos,eos); logits=model(x.unsqueeze(0),torch.zeros(1,11))
        vals.append(float(F.cross_entropy(logits.reshape(-1,logits.size(-1)),y,ignore_index=-100)))
    return sum(vals)/max(1,len(vals))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--base",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--steps",type=int,default=300); p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--rank",type=int,default=16); p.add_argument("--alpha",type=float,default=32.0); p.add_argument("--max-len",type=int,default=128); p.add_argument("--save-every",type=int,default=25); a=p.parse_args()
    random.seed(20260824); torch.manual_seed(20260824); torch.set_num_threads(min(2,torch.get_num_threads()))
    tok=Tokenizer.from_file(str(a.tokenizer)); pad=tok.token_to_id("<pad>") or 0; bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2
    positions=offsets(a.data); order=list(range(len(positions))); random.shuffle(order); val=order[:max(1,len(order)//10)]; train=order[len(val):]
    model=Voz(vocab_size=tok.get_vocab_size(),d_model=512,nhead=8,num_layers=8,dim_feedforward=2048,max_len=a.max_len,pad_token_id=pad,bos_token_id=bos,eos_token_id=eos)
    load_checkpoint(a.base,model=model,map_location="cpu"); total=sum(x.numel() for x in model.parameters()); trainable=treinar_lora_voz.add_lora(model,a.rank,a.alpha,0.05)
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr); a.out.mkdir(parents=True,exist_ok=True); latest=a.out/"latest.pt"; best=a.out/"best.pt"; best_val=float("inf"); gov=ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,ram_limit_percent=90)); rng=random.Random(20260824)
    print(json.dumps({"event":"chat_sft_initialized","examples":len(positions),"total_parameters":total,"trainable_parameters":trainable,"rank":a.rank,"target":"response+im_end+end_of_text+eos"},ensure_ascii=False),flush=True)
    for step in range(1,a.steps+1):
        x,y=read_example(a.data,positions[train[rng.randrange(len(train))]],tok,a.max_len,bos,eos); treinar_lora_voz.set_lora_train_mode(model); opt.zero_grad(set_to_none=True); logits=model(x.unsqueeze(0),torch.zeros(1,11)); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y,ignore_index=-100); loss.backward(); torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.0); opt.step()
        if step%a.save_every==0 or step==a.steps:
            vl=evaluate(model,a.data,[positions[i] for i in val],tok,a.max_len,bos,eos); metrics={"train_loss":float(loss.detach()),"val_loss":vl,"dataset":"human_chats_sft_curado","total_parameters":total,"trainable_parameters":trainable,"rank":a.rank,"target":"response+terminators","base":str(a.base)}; save_checkpoint(latest,model=model,optimizer=opt,step=step,metrics=metrics)
            if vl<best_val: best_val=vl; save_checkpoint(best,model=model,optimizer=opt,step=step,metrics=metrics)
            print(json.dumps({"step":step,**metrics,"best_val":best_val,**gov.throttle()},ensure_ascii=False),flush=True)
    print(f"[OK] Chat SFT concluído; melhor validação {best_val:.5f}",flush=True)


if __name__=="__main__": main()

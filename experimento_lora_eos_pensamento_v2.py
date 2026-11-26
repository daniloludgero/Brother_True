"""Versão compatível: LoRA no feed-forward e lm_head, sem envolver MHA."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from brothertrue.experimento_lora_eos_pensamento import LoRALinear, read_rows, batch, load_model, logits, evaluate


def inject_ffn(model):
    for layer in model.decoder.layers:
        layer.linear1 = LoRALinear(layer.linear1, 8, 16)
        layer.linear2 = LoRALinear(layer.linear2, 8, 16)
    model.lm_head = LoRALinear(model.lm_head, 8, 16)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--rows",type=int,default=1024)
    ap.add_argument("--steps",type=int,default=50); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--eos-weight",type=float,default=4.0)
    args=ap.parse_args(); random.seed(20260827); torch.manual_seed(20260827); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    args.out.mkdir(parents=True,exist_ok=True); rows=read_rows(args.data,args.rows); cut=max(8,int(len(rows)*.8)); train,val=rows[:cut],rows[cut:]
    tok=Tokenizer.from_file(str(args.tokenizer)); eos=tok.token_to_id("<eos>") or 2; model=load_model(args.base,tok.get_vocab_size()); inject_ffn(model)
    for p in model.parameters(): p.requires_grad=False
    trainable=[p for n,p in model.named_parameters() if "lora_" in n]
    for p in trainable: p.requires_grad_(True)
    xv,yv=batch(tok,val,256); before,eos_before,eos_count=evaluate(model,xv,yv,eos); opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01)
    for step in range(1,args.steps+1):
        start=((step-1)*8)%len(train); chosen=[train[(start+j)%len(train)] for j in range(8)]; x,y=batch(tok,chosen,256)
        model.train(); opt.zero_grad(set_to_none=True); fl=logits(model,x).flatten(0,1); fy=y.flatten(); raw=F.cross_entropy(fl,fy,ignore_index=-100,reduction="none")
        valid=fy!=-100; weights=torch.ones_like(raw); weights[fy==eos]=args.eos_weight; loss=(raw[valid]*weights[valid]).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if step==1 or step==args.steps or step%25==0: print(f"step={step}/{args.steps} weighted_loss={loss.item():.6f}",flush=True)
    after,eos_after,_=evaluate(model,xv,yv,eos); adapter={n:p.detach().cpu() for n,p in model.state_dict().items() if "lora_" in n}
    result={"experiment":"pensamento_lora_eos_window256_v2","rows":len(rows),"train_rows":len(train),"val_rows":len(val),"steps":args.steps,"rank":8,"alpha":16,"targets":["decoder.linear1","decoder.linear2","lm_head"],"eos_weight":args.eos_weight,"before_masked_loss":before,"after_masked_loss":after,"delta_after_minus_before":after-before,"eos_accuracy_before":eos_before,"eos_accuracy_after":eos_after,"eos_targets_in_validation":eos_count,"trainable_parameters":sum(p.numel() for p in trainable),"productive_by_masked_loss":after<before,"teacher_approved":False,"production_checkpoint_untouched":True}
    torch.save({"adapter_state_dict":adapter,"config":{"rank":8,"alpha":16,"context_window":256,"base_checkpoint":str(args.base)},"metrics":result},args.out/"adapter.pt")
    (args.out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()

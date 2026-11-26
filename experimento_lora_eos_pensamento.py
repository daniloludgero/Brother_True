"""Piloto LoRA + reforço de EOS para o pensamento com janela 256."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from pensamento_linguistico import PensamentoLinguistico


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=.05):
        super().__init__(); self.base = base
        self.rank, self.scale = rank, alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**.5)
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters(): p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.drop(self.lora_A(x))) * self.scale


def inject(module, rank=8, alpha=16):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject(child, rank, alpha)


def read_rows(path, limit):
    rows=[]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r=json.loads(line)
                if r.get("prompt") and r.get("think"): rows.append(r)
                if len(rows)>=limit: break
    if len(rows)<16: raise RuntimeError(f"Exemplos insuficientes: {len(rows)}")
    return rows


def encode(tok, row, max_len):
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    prefix=list(tok.encode(row["prompt"]+"\n<think>\n").ids)
    target=list(tok.encode(row["think"]).ids)
    prefix=([bos] if bos is not None else [])+prefix
    eos_ids=[eos] if eos is not None else []
    room=max_len+1-len(prefix)-len(eos_ids)
    if room < 2:
        prefix=prefix[:max_len-3]; room=max_len+1-len(prefix)-len(eos_ids)
    target=target[:max(1, room)]
    ids=prefix+target+eos_ids
    target_start=len(prefix)
    x,y=torch.tensor(ids[:-1],dtype=torch.long),torch.tensor(ids[1:],dtype=torch.long)
    # Ignora o prompt; aprende apenas think e EOS.
    y[:max(0,target_start-1)]=-100
    return x,y


def batch(tok, rows, max_len):
    pairs=[encode(tok,r,max_len) for r in rows]; width=max(len(x) for x,_ in pairs)
    pad=tok.token_to_id("<pad>") or 0; xs=[]; ys=[]
    for x,y in pairs:
        xs.append(F.pad(x,(0,width-len(x)),value=pad)); ys.append(F.pad(y,(0,width-len(y)),value=-100))
    return torch.stack(xs).long(),torch.stack(ys).long()


def load_model(path,vocab):
    m=PensamentoLinguistico(vocab_size=vocab,d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=256)
    p=torch.load(path,map_location="cpu",weights_only=False); state=p.get("model_state_dict",p.get("model",p))
    m.load_state_dict(state,strict=False); return m


def logits(model,x):
    return model(x,torch.zeros((x.size(0),11),dtype=torch.float32))


@torch.no_grad()
def evaluate(model,x,y,eos):
    model.eval(); flat_y=y.flatten(); flat_l=logits(model,x).flatten(0,1)
    normal=F.cross_entropy(flat_l,flat_y,ignore_index=-100)
    mask=(flat_y==eos)
    eos_acc=(flat_l.argmax(-1)[mask]==eos).float().mean() if mask.any() else torch.tensor(float("nan"))
    return float(normal), float(eos_acc), int(mask.sum())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--rows",type=int,default=1024)
    ap.add_argument("--steps",type=int,default=50); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--eos-weight",type=float,default=4.0)
    args=ap.parse_args(); random.seed(20260827); torch.manual_seed(20260827); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    args.out.mkdir(parents=True,exist_ok=True); rows=read_rows(args.data,args.rows); cut=max(8,int(len(rows)*.8)); train,val=rows[:cut],rows[cut:]
    tok=Tokenizer.from_file(str(args.tokenizer)); eos=tok.token_to_id("<eos>") or 2
    model=load_model(args.base,tok.get_vocab_size()); inject(model.decoder,8,16); model.lm_head=LoRALinear(model.lm_head,8,16)
    for p in model.parameters(): p.requires_grad=False
    trainable=[p for n,p in model.named_parameters() if "lora_" in n]; [p.requires_grad_(True) for p in trainable]
    xv,yv=batch(tok,val,256); before,eos_before,eos_count=evaluate(model,xv,yv,eos)
    opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01)
    for step in range(1,args.steps+1):
        start=((step-1)*8)%len(train); chosen=[train[(start+j)%len(train)] for j in range(8)]; x,y=batch(tok,chosen,256)
        model.train(); opt.zero_grad(set_to_none=True); fl=logits(model,x).flatten(0,1); fy=y.flatten()
        raw=F.cross_entropy(fl,fy,ignore_index=-100,reduction="none"); valid=fy!=-100; weights=torch.ones_like(raw); weights[fy==eos]=args.eos_weight
        loss=(raw[valid]*weights[valid]).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if step==1 or step==args.steps or step%25==0: print(f"step={step}/{args.steps} weighted_loss={loss.item():.6f}",flush=True)
    after,eos_after,_=evaluate(model,xv,yv,eos)
    adapter={n:p.detach().cpu() for n,p in model.state_dict().items() if "lora_" in n}
    result={"experiment":"pensamento_lora_eos_window256","rows":len(rows),"train_rows":len(train),"val_rows":len(val),"steps":args.steps,"rank":8,"alpha":16,"eos_weight":args.eos_weight,"before_masked_loss":before,"after_masked_loss":after,"delta_after_minus_before":after-before,"eos_accuracy_before":eos_before,"eos_accuracy_after":eos_after,"eos_targets_in_validation":eos_count,"trainable_parameters":sum(p.numel() for p in trainable),"productive_by_masked_loss":after<before,"teacher_approved":False,"production_checkpoint_untouched":True}
    torch.save({"adapter_state_dict":adapter,"config":{"rank":8,"alpha":16,"context_window":256,"base_checkpoint":str(args.base)},"metrics":result},args.out/"adapter.pt")
    (args.out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()

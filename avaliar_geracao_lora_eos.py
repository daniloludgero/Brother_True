"""Avalia geração do piloto 256 contra o mesmo modelo com adaptador LoRA."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from brothertrue.pensamento_linguistico import PensamentoLinguistico
from brothertrue.experimento_lora_eos_pensamento import LoRALinear


def load(path, vocab):
    m=PensamentoLinguistico(vocab_size=vocab,d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=256)
    p=torch.load(path,map_location="cpu",weights_only=False); m.load_state_dict(p.get("model_state_dict",p.get("model",p)),strict=False); return m.eval()


def inject(m):
    for layer in m.decoder.layers:
        layer.linear1=LoRALinear(layer.linear1,8,16); layer.linear2=LoRALinear(layer.linear2,8,16)
    m.lm_head=LoRALinear(m.lm_head,8,16)


def generate(m,tok,prompt,max_new):
    raw=list(tok.encode(prompt+"\n<think>\n").ids); ids=[m.bos_token_id]+raw[:m.max_len-max_new-1]; prefix=len(ids); humor=torch.zeros((1,11))
    for _ in range(max_new):
        with torch.no_grad(): nxt=int(torch.argmax(m(torch.tensor([ids]),humor)[0,-1]).item())
        ids.append(nxt)
        if nxt==m.eos_token_id: break
    return ids[prefix:]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,required=True); ap.add_argument("--base",type=Path,required=True); ap.add_argument("--adapter",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--samples",type=int,default=4); ap.add_argument("--max-new",type=int,default=24); args=ap.parse_args()
    rows=[]
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r=json.loads(line)
                if r.get("prompt") and r.get("think"): rows.append(r)
                if len(rows)>=args.samples: break
    tok=Tokenizer.from_file(str(args.tokenizer)); base=load(args.base,tok.get_vocab_size()); pilot=load(args.base,tok.get_vocab_size()); inject(pilot)
    p=torch.load(args.adapter,map_location="cpu",weights_only=False); pilot.load_state_dict(p["adapter_state_dict"],strict=False); pilot.eval()
    report={"experiment":"pensamento_lora_eos_generation","samples":len(rows),"max_new":args.max_new,"models":{}}
    for name,m in {"base":base,"lora_eos":pilot}.items():
        vals=[]
        for row in rows:
            ids=generate(m,tok,row["prompt"],args.max_new); text=tok.decode(ids,skip_special_tokens=True).strip(); target=set(tok.encode(row["think"]).ids)
            vals.append({"tokens":len(ids),"nonempty":bool(text),"eos":m.eos_token_id in ids,"target_token_overlap":len(set(ids)&target)/max(1,len(set(ids))),"adjacent_repeat_ratio":sum(a==b for a,b in zip(ids,ids[1:]))/max(1,len(ids)-1)})
        report["models"][name]={"nonempty_rate":sum(v["nonempty"] for v in vals)/len(vals),"eos_rate":sum(v["eos"] for v in vals)/len(vals),"mean_tokens":sum(v["tokens"] for v in vals)/len(vals),"mean_target_token_overlap":sum(v["target_token_overlap"] for v in vals)/len(vals),"mean_adjacent_repeat_ratio":sum(v["adjacent_repeat_ratio"] for v in vals)/len(vals),"samples":vals}
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__": main()

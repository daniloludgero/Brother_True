"""Sonda única de 8192 tokens, sem geração iterativa nem gradientes."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pensamento_linguistico import PensamentoLinguistico


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,required=True); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    torch.set_num_threads(2); tok=Tokenizer.from_file(str(args.tokenizer)); p=torch.load(args.base,map_location="cpu",weights_only=False); state=p.get("model_state_dict",p.get("model",p))
    m=PensamentoLinguistico(vocab_size=tok.get_vocab_size(),d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=8192)
    core={k:v for k,v in state.items() if k!="pos_embedding.weight"}; m.load_state_dict(core,strict=False)
    old=state.get("pos_embedding.weight")
    with torch.no_grad(): m.pos_embedding.weight.copy_(F.interpolate(old.T.unsqueeze(0),size=8192,mode="linear",align_corners=False).squeeze(0).T)
    corpus=[]
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r=json.loads(line)
                if r.get("prompt") and r.get("think"): corpus.extend(tok.encode(r["prompt"]+"\n<think>\n"+r["think"]).ids)
                if len(corpus)>30000: break
    ids=(([tok.token_to_id("<bos>") or 1]+corpus)*2)[:8192]; started=time.perf_counter(); result={"context_window":8192}
    try:
        with torch.no_grad(): out=m(torch.tensor([ids],dtype=torch.long),torch.zeros((1,11))); logits=out[0,-1]
        result.update({"status":"ok","elapsed_s":time.perf_counter()-started,"finite":bool(torch.isfinite(logits).all()),"entropy":float(-(F.softmax(logits,dim=-1)*F.log_softmax(logits,dim=-1)).sum().item())})
    except (RuntimeError,MemoryError) as exc: result.update({"status":"failed","elapsed_s":time.perf_counter()-started,"error":str(exc)[:500]})
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False),flush=True); del m; gc.collect()


if __name__=="__main__": main()

"""Teste top-down de capacidade contextual, sem treinamento."""
from __future__ import annotations
import argparse, gc, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pensamento_linguistico import PensamentoLinguistico


def memory_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def load_state(path):
    p=torch.load(path,map_location="cpu",weights_only=False)
    return p.get("model_state_dict",p.get("model",p))


def model_for(state,vocab,length):
    m=PensamentoLinguistico(vocab_size=vocab,d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=length)
    core={k:v for k,v in state.items() if k!="pos_embedding.weight"}; m.load_state_dict(core,strict=False)
    old=state.get("pos_embedding.weight")
    if old is not None:
        with torch.no_grad(): m.pos_embedding.weight.copy_(F.interpolate(old.T.unsqueeze(0),size=length,mode="linear",align_corners=False).squeeze(0).T)
    return m.eval()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,required=True); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    torch.set_num_threads(2); tok=Tokenizer.from_file(str(args.tokenizer)); state=load_state(args.base)
    corpus=[]
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r=json.loads(line)
                if r.get("prompt") and r.get("think"): corpus.extend(tok.encode(r["prompt"]+"\n<think>\n"+r["think"]).ids)
                if len(corpus)>20000: break
    bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2; corpus=[bos]+corpus
    lengths=[4096,3072,2048,1536,1024,768,512,256]; report={"experiment":"context_window_topdown_inference","candidates":[]}
    for length in lengths:
        before=memory_gb(); started=time.perf_counter(); record={"context_window":length,"ram_before_gb":before}
        try:
            m=model_for(state,tok.get_vocab_size(),length); context=(corpus*((length//len(corpus))+2))[:length-4]
            x=torch.tensor([context],dtype=torch.long); humor=torch.zeros((1,11));
            with torch.no_grad(): out=m(x,humor); logits=out[0,-1]; probs=F.softmax(logits,dim=-1); entropy=float(-(probs*torch.log(probs.clamp_min(1e-12))).sum().item()); first=int(torch.argmax(logits).item())
            generated=[first]; ids=context+[first]
            for _ in range(3):
                with torch.no_grad(): nxt=int(torch.argmax(m(torch.tensor([ids],dtype=torch.long),humor)[0,-1]).item())
                generated.append(nxt); ids.append(nxt)
            record.update({"status":"ok","elapsed_s":time.perf_counter()-started,"ram_after_gb":memory_gb(),"finite":bool(torch.isfinite(logits).all()),"next_token":first,"entropy":entropy,"generated_tokens":len(generated),"eos_in_generated":eos in generated,"adjacent_repeat_ratio":sum(a==b for a,b in zip(generated,generated[1:]))/3})
        except (RuntimeError,MemoryError) as exc:
            record.update({"status":"failed","elapsed_s":time.perf_counter()-started,"ram_after_gb":memory_gb(),"error":str(exc)[:500]})
        report["candidates"].append(record); print(json.dumps(record,ensure_ascii=False),flush=True)
        try: del m
        except UnboundLocalError: pass
        gc.collect()
    ok=[r for r in report["candidates"] if r["status"]=="ok"]; report["max_successful_window"] = max((r["context_window"] for r in ok),default=None)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"max_successful_window":report["max_successful_window"],"ok_windows":[r["context_window"] for r in ok]},ensure_ascii=False),flush=True)


if __name__=="__main__": main()

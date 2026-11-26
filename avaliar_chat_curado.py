"""Avalia o SFT curado usando terminadores de conversa."""

from __future__ import annotations

import argparse, json, random, math
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from . import treinar_lora_voz, treinar_sft_chat_lora
from .checkpoint_manager_safe import load_checkpoint
from .treino_voz_texto_limpo import offsets

treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)


@torch.no_grad()
def generate(model, tok, prompt, max_len=128, max_new=48):
    bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2
    stop=tok.encode("<|im_end|><|end_of_text|>",add_special_tokens=False).ids
    pids=tok.encode(prompt,add_special_tokens=False).ids[-(max_len-max_new-1):]
    ids=[bos]+pids; gen=[]; x=torch.tensor([ids])
    for _ in range(max_new):
        nxt=int(model(x,torch.zeros(1,11))[0,-1].argmax()); gen.append(nxt); x=torch.cat([x,torch.tensor([[nxt]])],1)
        if nxt==eos or (len(gen)>=len(stop) and gen[-len(stop):]==stop) or x.size(1)>=max_len: break
    return {"completion":tok.decode(gen,skip_special_tokens=True),"new_tokens":len(gen),"ended_eos":bool(gen and gen[-1]==eos),"ended_chat_marker":len(gen)>=len(stop) and gen[-len(stop):]==stop,"unique_ratio":len(set(gen))/max(1,len(gen))}


def load_adapter(tok, base, adapter, max_len):
    model=treinar_sft_chat_lora.Voz(vocab_size=tok.get_vocab_size(),d_model=512,nhead=8,num_layers=8,dim_feedforward=2048,max_len=max_len,pad_token_id=tok.token_to_id("<pad>") or 0,bos_token_id=tok.token_to_id("<bos>") or 1,eos_token_id=tok.token_to_id("<eos>") or 2)
    load_checkpoint(base,model=model,map_location="cpu"); treinar_lora_voz.add_lora(model,16,32,0.05); payload=torch.load(adapter,map_location="cpu",weights_only=False); model.load_state_dict(payload["adapter_state_dict"],strict=False); model.eval(); return model


@torch.no_grad()
def loss(model,data,positions,tok,max_len):
    bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2; vals=[]
    for pos in positions:
        x,y=treinar_sft_chat_lora.read_example(data,pos,tok,max_len,bos,eos); z=model(x.unsqueeze(0),torch.zeros(1,11)); vals.append(float(F.cross_entropy(z.reshape(-1,z.size(-1)),y,ignore_index=-100)))
    v=sum(vals)/max(1,len(vals)); return {"loss":v,"perplexity":math.exp(min(v,20)),"samples":len(vals)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--base",type=Path,required=True); p.add_argument("--adapter",type=Path,required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--max-len",type=int,default=128); a=p.parse_args()
    torch.set_num_threads(min(2,torch.get_num_threads())); tok=Tokenizer.from_file(str(a.tokenizer)); allpos=offsets(a.data); order=list(range(len(allpos))); random.Random(20260824).shuffle(order); positions=[allpos[i] for i in order[:8]]
    model=load_adapter(tok,a.base,a.adapter,a.max_len); examples=[]
    for pos in positions:
        with a.data.open("rb") as h: h.seek(pos); item=json.loads(h.readline().decode("utf-8"))
        examples.append({"reference":item["response"],"generated":generate(model,tok,item["prompt"],a.max_len,48)})
    result={"model":"Voz 50M chat SFT curado","response_loss":loss(model,a.data,positions,tok,a.max_len),"examples":examples}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()

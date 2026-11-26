"""Avalia geração com tokenizer de chat e tokens especiais dedicados."""

from __future__ import annotations
import argparse, json, random, math
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from . import treinar_lora_voz, treinar_sft_chat_special
from .checkpoint_manager_safe import load_checkpoint
from .treino_voz_texto_limpo import offsets

treinar_lora_voz.LoRALinear.weight=property(lambda self:self.base.weight)
treinar_lora_voz.LoRALinear.bias=property(lambda self:self.base.bias)


def load_model(tok,base,adapter,max_len):
    model=treinar_sft_chat_special.expanded_base(base,16384,tok.get_vocab_size(),tok,max_len); treinar_lora_voz.add_lora(model,16,32,0.05); payload=torch.load(adapter,map_location="cpu",weights_only=False); model.load_state_dict(payload["adapter_state_dict"],strict=False); model.eval(); return model


@torch.no_grad()
def generate(model,tok,prompt,max_len=128,max_new=48):
    bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2; im_end=tok.token_to_id("<|im_end|>"); end_text=tok.token_to_id("<|end_of_text|>"); p=tok.encode(prompt,add_special_tokens=False).ids[-(max_len-max_new-1):]; ids=[bos]+p; gen=[]; x=torch.tensor([ids])
    for _ in range(max_new):
        nxt=int(model(x,torch.zeros(1,11))[0,-1].argmax()); gen.append(nxt); x=torch.cat([x,torch.tensor([[nxt]])],1)
        if nxt==eos or nxt==im_end or nxt==end_text or x.size(1)>=max_len: break
    return {"completion":tok.decode(gen,skip_special_tokens=True),"new_tokens":len(gen),"ended_eos":bool(gen and gen[-1]==eos),"ended_im_end":bool(gen and gen[-1]==im_end),"unique_ratio":len(set(gen))/max(1,len(gen))}


@torch.no_grad()
def loss(model,data,positions,tok,max_len):
    bos=tok.token_to_id("<bos>") or 1; eos=tok.token_to_id("<eos>") or 2; vals=[]
    for pos in positions:
        x,y=treinar_sft_chat_special.read_example(data,pos,tok,max_len,bos,eos); z=model(x.unsqueeze(0),torch.zeros(1,11)); vals.append(float(F.cross_entropy(z.reshape(-1,z.size(-1)),y,ignore_index=-100)))
    v=sum(vals)/max(1,len(vals)); return {"loss":v,"perplexity":math.exp(min(v,20)),"samples":len(vals)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--base",type=Path,required=True); p.add_argument("--adapter",type=Path,required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--max-len",type=int,default=128); a=p.parse_args()
    torch.set_num_threads(min(2,torch.get_num_threads())); tok=Tokenizer.from_file(str(a.tokenizer)); allpos=offsets(a.data); order=list(range(len(allpos))); random.Random(20260824).shuffle(order); positions=[allpos[i] for i in order[:8]]; model=load_model(tok,a.base,a.adapter,a.max_len); examples=[]
    for pos in positions:
        with a.data.open("rb") as h: h.seek(pos); item=json.loads(h.readline().decode("utf-8"))
        examples.append({"reference":item["response"],"generated":generate(model,tok,item["prompt"],a.max_len,48)})
    result={"model":"Voz 50M chat special","vocab_size":tok.get_vocab_size(),"special_ids":{x:tok.token_to_id(x) for x in ["<|im_start|>","<|im_end|>","<|end_of_text|>"]},"response_loss":loss(model,a.data,positions,tok,a.max_len),"examples":examples}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()

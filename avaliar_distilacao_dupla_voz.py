"""Avaliação curta do adaptador da Voz treinado com dois professores."""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from . import treinar_lora_voz
from .checkpoint_manager_safe import load_checkpoint
from .treinar_distilacao_dupla_voz import expanded_base, read_example

treinar_lora_voz.LoRALinear.weight=property(lambda self:self.base.weight)
treinar_lora_voz.LoRALinear.bias=property(lambda self:self.base.bias)

@torch.no_grad()
def generate(model,tok,prompt,max_len=128,max_new=48):
    bos=tok.token_to_id('<bos>') or 1; eos=tok.token_to_id('<eos>') or 2; end=tok.token_to_id('<|im_end|>'); final=tok.token_to_id('<|end_of_text|>')
    ids=[bos]+tok.encode(prompt,add_special_tokens=False).ids[-(max_len-max_new-1):]; x=torch.tensor([ids]); out=[]
    for _ in range(max_new):
        nxt=int(model(x,torch.zeros(1,11))[0,-1].argmax()); out.append(nxt); x=torch.cat([x,torch.tensor([[nxt]])],1)
        if nxt in (eos,end,final) or x.size(1)>=max_len: break
    return {'completion':tok.decode(out,skip_special_tokens=True),'new_tokens':len(out),'ended_eos':bool(out and out[-1]==eos),'unique_ratio':len(set(out))/max(1,len(out))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--base',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2); tok=Tokenizer.from_file(str(a.tokenizer)); model=expanded_base(a.base,16384,tok.get_vocab_size(),tok,128); treinar_lora_voz.add_lora(model,8,16,0.05); payload=load_checkpoint(a.adapter,model=model,map_location='cpu'); model.eval()
    rows=[json.loads(x) for x in a.data.read_text(encoding='utf-8').splitlines() if x.strip()]; examples=[]
    for row in rows[:2]+rows[-2:]: examples.append({'complexity':row['complexity'],'teacher':row['teacher'],'generated':generate(model,tok,row['prompt'])})
    result={'checkpoint_step':payload.get('step',80),'metrics':payload.get('metrics',{}),'examples':examples,'dataset_rows':len(rows)}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()

from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from . import treino_lora_multitarefa_continuidade_v2 as treino

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); tok=Tokenizer.from_file(str(a.tokenizer)); eos=tok.token_to_id('<eos>') or 2
    rows=treino.read_rows(a.data, 48); cut=max(8,int(len(rows)*.8)); _, val=rows[:cut],rows[cut:]; x,y,seg=treino.make_batch(tok,val)
    base=treino.load_model(a.base,tok.get_vocab_size()); pilot=treino.load_model(a.base,tok.get_vocab_size()); treino.inject(pilot)
    payload=torch.load(a.adapter,map_location='cpu',weights_only=False); pilot.load_state_dict(payload['adapter_state_dict'],strict=False)
    result={'eos_id':eos,'val_rows':len(val),'models':{}}
    for name,model in {'base':base,'lora_graphrag_eos':pilot}.items():
        model.eval(); logits=treino.forward(model,x).flatten(0,1); mask=(seg.flatten()==4)&(y.flatten()!=-100); selected=logits[mask]; target=selected[:,eos]; ranks=(selected>target[:,None]).sum(1)+1; probs=torch.softmax(selected,dim=-1)[:,eos]
        result['models'][name]={'positions':int(mask.sum()),'mean_eos_probability':float(probs.mean()),'mean_eos_rank':float(ranks.float().mean()),'median_eos_rank':float(ranks.float().median()),'top1_rate':float((ranks==1).float().mean()),'min_rank':int(ranks.min()),'max_rank':int(ranks.max())}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from . import treino_lora_multitarefa_continuidade_v2 as t

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--rows',type=int,default=128); ap.add_argument('--adapter',action='append',default=[]); a=ap.parse_args(); torch.set_num_threads(2); tok=Tokenizer.from_file(str(a.tokenizer)); rows=t.read_rows(a.data,a.rows); x,y,s=t.make_batch(tok,rows); result={'experiment':'avaliacao_adaptadores_ampliado_n','data':str(a.data),'rows':len(rows),'eos_id':tok.token_to_id('<eos>') or 2,'models':{}}
    models={'base':None}; models.update(dict(spec.split('=',1) for spec in a.adapter))
    for name,path in models.items():
        m=t.load_model(a.base,tok.get_vocab_size())
        if path:
            t.inject(m); m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['adapter_state_dict'],strict=False)
        result['models'][name]=t.scores(m,x,y,s)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

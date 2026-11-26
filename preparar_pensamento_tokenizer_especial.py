from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .pensamento_linguistico import PensamentoLinguistico

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--tokenizer-out',type=Path,required=True); ap.add_argument('--checkpoint-out',type=Path,required=True); a=ap.parse_args()
    tok=Tokenizer.from_file(str(a.tokenizer)); old=tok.get_vocab_size(); markers=['<system>','<user>','<assistant>','<think>','</think>','<final>','</final>']; added=tok.add_special_tokens(markers); new=tok.get_vocab_size(); a.tokenizer_out.parent.mkdir(parents=True,exist_ok=True); tok.save(str(a.tokenizer_out))
    old_m=PensamentoLinguistico(vocab_size=old,d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=256); payload=torch.load(a.base,map_location='cpu',weights_only=False); old_m.load_state_dict(payload.get('model_state_dict',payload.get('model',payload)),strict=False)
    new_m=PensamentoLinguistico(vocab_size=new,d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=256); state=new_m.state_dict();
    for key,value in old_m.state_dict().items():
        if key in state and state[key].shape==value.shape: state[key]=value
        elif key=='token_embedding.weight': state[key][:old]=value
        elif key=='lm_head.weight': state[key][:old]=value
        elif key=='lm_head.bias': state[key][:old]=value
    new_m.load_state_dict(state); a.checkpoint_out.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':new_m.state_dict(),'config':{'vocab_size':new,'d_model':384,'nhead':6,'num_layers':4,'dim_feedforward':1536,'max_len':256},'base_checkpoint':str(a.base),'special_tokens':markers},a.checkpoint_out)
    print(json.dumps({'old_vocab':old,'new_vocab':new,'added':added,'tokenizer':str(a.tokenizer_out),'checkpoint':str(a.checkpoint_out),'original_untouched':True},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import argparse,json,re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .pensamento_linguistico import PensamentoLinguistico

QUESTIONS=['Explique em duas frases por que uma hipótese precisa ser testada.','Compare janela contextual e qualidade dos dados para um computador com pouca memória.','Responda somente com JSON válido contendo as chaves "final_answer" e "tool_calls".','What is the difference between correlation and causation? Give one concise example.']

@torch.no_grad()
def generate(m,tok,q,max_new):
    prompt='<system>\nYou are a helpful assistant. Answer clearly.\n<user>\n'+q+'\n<assistant>\n<think>\n'; ids=[m.bos_token_id]+tok.encode(prompt,add_special_tokens=False).ids; ids=ids[-(m.max_len-max_new):]; prefix=len(ids); x=torch.tensor([ids]); humor=torch.zeros(1,11); out=[]; eos=tok.token_to_id('<eos>') or 2
    for _ in range(max_new):
        n=int(m(x,humor)[0,-1].argmax()); out.append(n); x=torch.cat([x,torch.tensor([[n]])],1)
        if n==eos or n==tok.token_to_id('</think>') or n==tok.token_to_id('</final>') or x.size(1)>=m.max_len: break
    text=tok.decode(out,skip_special_tokens=True).strip(); public=re.sub(r'<think>.*?</think>','',text,flags=re.I|re.S); return {'prompt_tokens':prefix-1,'generated_tokens':len(out),'stop_token':tok.id_to_token(out[-1]) if out else None,'raw_text':text,'public_text':public.strip(),'nonempty':bool(public.strip())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-new',type=int,default=64); a=ap.parse_args(); tok=Tokenizer.from_file(str(a.tokenizer)); m=PensamentoLinguistico(vocab_size=tok.get_vocab_size(),d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=256); p=torch.load(a.checkpoint,map_location='cpu',weights_only=False); m.load_state_dict(p['model_state_dict']); m.eval(); rows=[{'query':q,**generate(m,tok,q,a.max_new)} for q in QUESTIONS]; report={'experiment':'pensamento_tokenizer_especial_isolado','vocab_size':tok.get_vocab_size(),'special_ids':{x:tok.token_to_id(x) for x in ['<system>','<user>','<assistant>','<think>','</think>','<final>','</final>','<eos>']},'questions':rows}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

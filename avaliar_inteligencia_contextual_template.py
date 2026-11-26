from __future__ import annotations
import argparse,json
from pathlib import Path
from .avaliar_inteligencia_contextual import QUESTIONS, load, generate
from tokenizers import Tokenizer
import torch

def format_prompt(question):
    return '<system>\nYou are a helpful assistant. Answer clearly, conservatively, and follow the requested format.\n<user>\n'+question+'\n<assistant>\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-new',type=int,default=64); a=ap.parse_args(); torch.set_num_threads(2); tok=Tokenizer.from_file(str(a.tokenizer)); base=load(a.base,tok.get_vocab_size()); lora=load(a.base,tok.get_vocab_size(),True); lora.load_state_dict(torch.load(a.adapter,map_location='cpu',weights_only=False)['adapter_state_dict'],strict=False); report={'experiment':'avaliacao_inteligencia_contextual_template','max_new':a.max_new,'format':'system-user-assistant-think','models':{'base':{},'lora_calibrado':{}}}
    for item in QUESTIONS:
        p=format_prompt(item['prompt'])
        for name,m in [('base',base),('lora_calibrado',lora)]: report['models'][name][item['id']]=generate(m,tok,p,a.max_new)
    for name in report['models']:
        vals=list(report['models'][name].values()); report['models'][name]['summary']={'nonempty_rate':sum(v['nonempty'] for v in vals)/len(vals),'eos_rate':sum(v['eos'] for v in vals)/len(vals),'mean_prompt_tokens':sum(v['prompt_tokens'] for v in vals)/len(vals),'mean_output_tokens':sum(v['output_tokens'] for v in vals)/len(vals),'mean_adjacent_repeat_ratio':sum(v['adjacent_repeat_ratio'] for v in vals)/len(vals),'mean_unique_token_ratio':sum(v['unique_token_ratio'] for v in vals)/len(vals)}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

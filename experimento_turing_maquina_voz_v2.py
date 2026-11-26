"""Correção do teste VT-1: evita recarregar pesos antigos sobre vocabulário expandido."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .treinar_distilacao_dupla_voz import expanded_base
from .experimento_turing_maquina_voz import protocol_prefix, generate, score

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--max-len',type=int,default=128); p.add_argument('--max-new',type=int,default=24); a=p.parse_args(); torch.set_num_threads(2); tok=Tokenizer.from_file(str(a.tokenizer)); model=expanded_base(a.base,16384,tok.get_vocab_size(),tok,a.max_len); model.eval(); rows=[]
    for turns in (1,2,3,4,5,6):
        ids,text=generate(model,tok,protocol_prefix(turns),a.max_len,a.max_new); rows.append({'turns':turns,'completion':text,'score':score(text,turns),'ended_special':bool(ids and ids[-1] in {tok.token_to_id('<eos>'),tok.token_to_id('<|im_end|>'),tok.token_to_id('<|end_of_text|>')})})
    report={'protocol':'VT-1 machine language','model':str(a.base),'interpretation':'upper-bound prefix-following test; public Voz.gerar does not accept text context','sequence':rows,'summary':{'max_matched':max(x['score']['matched'] for x in rows),'turns_tested':len(rows),'passed_any':any(x['score']['matched']>=2 for x in rows)}}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()

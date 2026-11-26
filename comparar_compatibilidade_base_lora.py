from __future__ import annotations
import argparse,json
from pathlib import Path
from .compatibilidade_geracao_experimental import GeradorContextualCompativel

QUESTIONS=['Explique em duas frases por que uma hipótese precisa ser testada.','Compare janela contextual e qualidade dos dados para um computador com pouca memória.','Responda somente com JSON válido contendo as chaves "final_answer" e "tool_calls".','What is the difference between correlation and causation? Give one concise example.']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); reports={}
    for name,adapter in [('base',None),('lora',a.adapter)]:
        g=GeradorContextualCompativel(a.base,a.tokenizer,adapter,128); reports[name]=[{'query':q,**g.responder(q,48)} for q in QUESTIONS]
    report={'experiment':'comparacao_compatibilidade_base_lora','reports':reports}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

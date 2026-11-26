from __future__ import annotations
import argparse,json
from pathlib import Path
from .compatibilidade_geracao_experimental import GeradorContextualCompativel

QUESTIONS=[
    'Explique em duas frases por que uma hipótese precisa ser testada.',
    'Compare janela contextual e qualidade dos dados para um computador com pouca memória.',
    'Responda somente com JSON válido contendo as chaves "final_answer" e "tool_calls".',
    'What is the difference between correlation and causation? Give one concise example.',
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-new',type=int,default=48); a=ap.parse_args(); gen=GeradorContextualCompativel(a.base,a.tokenizer,a.adapter,128); rows=[{'id':str(i+1),'query':q,**gen.responder(q,a.max_new)} for i,q in enumerate(QUESTIONS)]; report={'experiment':'compatibilidade_geracao_chat_special','architecture':'Voz 50M + tokenizer special + LoRA','questions':rows,'summary':{'nonempty_rate':sum(r['nonempty'] for r in rows)/len(rows),'eos_or_terminator_rate':sum(r['stop_reason'] in ('eos','im_end','end_of_text') for r in rows)/len(rows),'mean_prompt_tokens':sum(r['prompt_tokens'] for r in rows)/len(rows),'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/len(rows)}}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

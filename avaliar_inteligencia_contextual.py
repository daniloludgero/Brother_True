from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from . import treino_lora_multitarefa_continuidade_v2 as treino

QUESTIONS = [
    {"id":"abstracao_pt","prompt":"Explique por que uma cidade pode ser entendida como um sistema, usando uma analogia simples e indicando onde a analogia deixa de funcionar."},
    {"id":"composicional","prompt":"Há três caixas: A é maior que B, B é maior que C, mas C pesa mais que A. O que pode ser concluído com segurança? Separe fato, inferência e incerteza."},
    {"id":"ambiguidade","prompt":"A frase 'o técnico viu o paciente com o aparelho' é ambígua. Apresente duas interpretações e faça uma pergunta curta para desambiguar."},
    {"id":"tokenizacao","prompt":"Considere a sequência `Ax_17::beta/Δ`. Descreva quais propriedades um tokenizador deve preservar ao processá-la e não invente a segmentação exata."},
    {"id":"instrucao_estruturada","prompt":"Responda somente com um objeto JSON válido contendo as chaves 'hipotese', 'evidencia' e 'limite'. O tema é: por que medir uma hipótese antes de aceitá-la?"},
    {"id":"generalizacao","prompt":"Uma regra funcionou em quatro casos, mas falhou no quinto. Dê três explicações possíveis e proponha o teste mais informativo para distingui-las."},
    {"id":"tradeoff","prompt":"Compare aumentar a janela contextual e aumentar a qualidade dos dados. Dê uma recomendação para um computador com pouca memória, explicitando o compromisso."},
    {"id":"contradicao","prompt":"Se uma fonte afirma que X é impossível e outra demonstra um caso de X, como um sistema confiável deve responder antes de escolher uma das fontes?"},
    {"id":"english_transfer","prompt":"In one concise paragraph, explain the difference between correlation and causation, then give one example where correlation is misleading."},
    {"id":"contexto_longo","prompt":"Memorize estes fatos: o projeto usa LoRA; o checkpoint principal não deve ser alterado; o experimento atual é descartável; o objetivo é medir generalização. Qual ação é compatível com todos eles e por quê?"},
]

def load(path, vocab, inject=False):
    m=treino.load_model(path,vocab)
    if inject: treino.inject(m)
    return m

@torch.no_grad()
def generate(model, tok, prompt, max_new):
    raw=tok.encode(prompt+'\n<think>\n').ids; bos=tok.token_to_id('<bos>') or 1; eos=tok.token_to_id('<eos>') or 2
    ids=[bos]+raw[:model.max_len-max_new-1]; prefix=len(ids)
    humor=torch.zeros((1,11)); model.eval()
    for _ in range(max_new):
        nxt=int(torch.argmax(model(torch.tensor([ids]),humor)[0,-1]).item()); ids.append(nxt)
        if nxt==eos: break
    generated=ids[prefix:]; text=tok.decode(generated,skip_special_tokens=True).strip()
    repeats=sum(a==b for a,b in zip(generated,generated[1:]))/max(1,len(generated)-1)
    return {'prompt_tokens':len(raw),'output_tokens':len(generated),'text':text,'nonempty':bool(text),'eos':eos in generated,'adjacent_repeat_ratio':repeats,'unique_token_ratio':len(set(generated))/max(1,len(generated))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-new',type=int,default=64); a=ap.parse_args(); torch.set_num_threads(2)
    tok=Tokenizer.from_file(str(a.tokenizer)); base=load(a.base,tok.get_vocab_size()); lora=load(a.base,tok.get_vocab_size(),True); lora.load_state_dict(torch.load(a.adapter,map_location='cpu',weights_only=False)['adapter_state_dict'],strict=False)
    report={'experiment':'avaliacao_inteligencia_contextual_fora_distribuicao','max_new':a.max_new,'questions':QUESTIONS,'models':{'base':{},'lora_calibrado':{}}}
    for item in QUESTIONS:
        for name,m in [('base',base),('lora_calibrado',lora)]: report['models'][name][item['id']]=generate(m,tok,item['prompt'],a.max_new)
    for name in report['models']:
        vals=list(report['models'][name].values()); report['models'][name]['summary']={'nonempty_rate':sum(v['nonempty'] for v in vals)/len(vals),'eos_rate':sum(v['eos'] for v in vals)/len(vals),'mean_prompt_tokens':sum(v['prompt_tokens'] for v in vals)/len(vals),'mean_output_tokens':sum(v['output_tokens'] for v in vals)/len(vals),'mean_adjacent_repeat_ratio':sum(v['adjacent_repeat_ratio'] for v in vals)/len(vals),'mean_unique_token_ratio':sum(v['unique_token_ratio'] for v in vals)/len(vals)}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

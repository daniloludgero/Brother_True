"""Amplia e qualifica o conjunto roteado de dois professores."""

from __future__ import annotations

import argparse, json, random, re
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MESSAGE_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)
SYSTEM = ("Você é o professor de perguntas breves. Responda apenas ao último usuário, "
          "com linguagem natural, útil e concisa. Não mencione modelos, instruções internas "
          "ou ferramentas.")

def messages(prompt):
    rows=[{"role":r,"content":c.strip()} for r,c in MESSAGE_RE.findall(prompt) if c.strip()]
    return [{"role":"system","content":SYSTEM}]+rows

def brief_candidates(path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            ms=messages(row.get('prompt',''))
            if len(ms)<=1: continue
            user=ms[-1]['content'].lower(); words=user.split()
            if 2<=len(words)<=80 and not any(x in user for x in ('tool','function','repository','api schema','pharmac','medication')):
                rows.append(row)
    random.shuffle(rows); return rows

def valid_complex(raw):
    try: obj=json.loads(raw)
    except json.JSONDecodeError: return None
    final=str(obj.get('final_answer','')).strip(); low=final.lower()
    if not 8<=len(final.split())<=180: return None
    if any(x in low for x in ('toolingtooling','<commit_msg>','<commit_after>')): return None
    if low.count('exploit')>4 or low.count('tooling')>4: return None
    calls=obj.get('tool_calls',[])
    if not isinstance(calls,list): return None
    out=[]
    for call in calls:
        if not isinstance(call,dict) or not call.get('name'): return None
        args=call.get('arguments',{})
        if not isinstance(args,dict): return None
        out.append({'name':call['name'],'arguments':args})
    return json.dumps({'final_answer':final,'tool_calls':out},ensure_ascii=False,separators=(',',':'))

def complex_candidates(path, limit):
    rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['prompt','response','kind','difficulty']):
        for row in batch.to_pylist():
            if row.get('kind')!='success': continue
            target=valid_complex(row.get('response',''))
            if target and row.get('prompt'):
                rows.append((row,target))
        if len(rows)>=limit*2: break
    random.shuffle(rows); return rows[:limit]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--brief-data',type=Path,required=True); p.add_argument('--complex-data',type=Path,required=True); p.add_argument('--teacher',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--brief-limit',type=int,default=96); p.add_argument('--complex-limit',type=int,default=384); a=p.parse_args()
    random.seed(20260825); torch.set_num_threads(2); brief=brief_candidates(a.brief_data)[:a.brief_limit]; complex_rows=complex_candidates(a.complex_data,a.complex_limit)
    tok=AutoTokenizer.from_pretrained(str(a.teacher),local_files_only=True); model=AutoModelForCausalLM.from_pretrained(str(a.teacher),local_files_only=True,dtype=torch.float32,low_cpu_mem_usage=True); model.eval(); a.output.parent.mkdir(parents=True,exist_ok=True); a.manifest.parent.mkdir(parents=True,exist_ok=True); b=c=filtered=0
    with a.output.open('w',encoding='utf-8') as out:
        for row in brief:
            try:
                enc=tok.apply_chat_template(messages(row['prompt']),add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt'); ids=enc['input_ids'][:,-256:]; mask=enc['attention_mask'][:,-256:]
                with torch.inference_mode(): gen=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=48,do_sample=False,repetition_penalty=1.04,pad_token_id=tok.eos_token_id)
                answer=tok.decode(gen[0,ids.shape[-1]:],skip_special_tokens=True).strip(); low=answer.lower()
            except Exception: filtered+=1; continue
            if not 4<=len(answer.split())<=96 or any(x in low for x in ('as an ai language model','i am a large language model','i apologize','i’m sorry','i am sorry')) or len(set(answer.split()))/max(1,len(answer.split()))<0.35:
                filtered+=1; continue
            out.write(json.dumps({'module':'voz','regime':'two_teacher_distillation_qualified','complexity':'brief','teacher':'qwen_base_brief','prompt':row['prompt'],'response':answer,'target_suffix':'<|im_end|><|end_of_text|>'},ensure_ascii=False)+'\n'); b+=1
            if b%16==0: print(json.dumps({'event':'brief','written':b},ensure_ascii=False),flush=True)
        for row,target in complex_rows:
            prompt='<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
            out.write(json.dumps({'module':'voz','regime':'two_teacher_distillation_qualified','complexity':'complex','teacher':'deepseek_agentic_teacher_canonical','prompt':prompt,'response':target,'target_suffix':'<|im_end|><|end_of_text|>','difficulty':row.get('difficulty','')},ensure_ascii=False)+'\n'); c+=1
    result={'brief_candidates':len(brief),'brief_written':b,'complex_written':c,'filtered':filtered,'total':b+c,'output':str(a.output),'quality':'schema,length,boilerplate,repetition'}; a.manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()

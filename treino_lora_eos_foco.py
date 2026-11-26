from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from . import treino_lora_multitarefa_continuidade_v2 as t

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--rows',type=int,default=48); ap.add_argument('--steps',type=int,default=120)
    a=ap.parse_args(); random.seed(20260829); torch.manual_seed(20260829); torch.set_num_threads(2); a.out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(str(a.tokenizer)); eos=tok.token_to_id('<eos>') or 2; rows=t.read_rows(a.data,a.rows); cut=max(8,int(len(rows)*.8)); train,val=rows[:cut],rows[cut:]
    model=t.load_model(a.base,tok.get_vocab_size()); t.inject(model)
    for p in model.parameters(): p.requires_grad=False
    trainable=[p for n,p in model.named_parameters() if 'lora_' in n]
    for p in trainable: p.requires_grad_(True)
    xv,yv,sv=t.make_batch(tok,val)
    def eval_eos():
        model.eval(); logits=t.forward(model,xv).flatten(0,1); yy=yv.flatten(); ss=sv.flatten(); mask=(ss==4)&(yy!=-100); z=logits[mask]; loss=F.cross_entropy(z,yy[mask]); rank=((z>z[:,eos,None]).sum(1)+1); return {'loss':float(loss.detach()),'accuracy':float((z.argmax(-1)==eos).float().mean()),'mean_probability':float(torch.softmax(z,dim=-1)[:,eos].mean().detach()),'median_rank':float(rank.float().median()),'mean_rank':float(rank.float().mean())}
    before=eval_eos(); opt=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=.01)
    for step in range(1,a.steps+1):
        start=((step-1)*4)%len(train); chosen=[train[(start+j)%len(train)] for j in range(4)]; x,y,s=t.make_batch(tok,chosen); flat_y=y.flatten(); flat_s=s.flatten(); logits=t.forward(model,x).flatten(0,1); mask=(flat_s==4)&(flat_y!=-100); loss=F.cross_entropy(logits[mask],flat_y[mask]); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if step==1 or step==a.steps or step%30==0: print(f'step={step}/{a.steps} eos_loss={loss.item():.6f}',flush=True)
    after=eval_eos(); adapter={n:p.detach().cpu() for n,p in model.state_dict().items() if 'lora_' in n}; result={'experiment':'pensamento_lora_eos_foco','data':str(a.data),'base_checkpoint':str(a.base),'rows':len(rows),'train_rows':len(train),'val_rows':len(val),'steps':a.steps,'eos_id':eos,'before':before,'after':after,'trainable_parameters':sum(p.numel() for p in trainable),'production_checkpoint_untouched':True,'storage_root':'D:\\brothertrue'}
    torch.save({'adapter_state_dict':adapter,'config':{'rank':8,'alpha':16,'context_window':256,'base_checkpoint':str(a.base),'targets':['decoder.linear1','decoder.linear2','lm_head']},'metrics':result},a.out/'adapter.pt'); (a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with Path(r'D:\brothertrue\training_memory_journal.jsonl').open('a',encoding='utf-8') as h: h.write(json.dumps({'event':'eos_focused_training','result_path':str(a.out/'result.json'),'adapter_path':str(a.out/'adapter.pt'),**result},ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()

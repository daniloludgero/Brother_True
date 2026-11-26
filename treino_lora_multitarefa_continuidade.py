"""Treino isolado: prompt, pensamento, resposta final e EOS em perdas separadas."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from pensamento_linguistico import PensamentoLinguistico


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__(); self.base=base; self.scale=alpha/rank
        self.lora_A=nn.Linear(base.in_features,rank,bias=False); self.lora_B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight,a=5**.5); nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters(): p.requires_grad=False
    def forward(self,x): return self.base(x)+self.lora_B(self.lora_A(x))*self.scale


def inject(model):
    for layer in model.decoder.layers:
        layer.linear1=LoRALinear(layer.linear1); layer.linear2=LoRALinear(layer.linear2)
    model.lm_head=LoRALinear(model.lm_head)


def keep_edges(ids, limit):
    if len(ids)<=limit: return ids
    left=max(1,limit//3); return ids[:left]+ids[-(limit-left):]


def good(row):
    q=row.get('quality_features',{}); s=float(row.get('quality_score_proxy',0))
    return bool(row.get('prompt','').strip() and row.get('think','').strip() and row.get('final','').strip()
        and '<think>' in row.get('prompt','') and q.get('reasoning_structure',0)>=1
        and q.get('completeness',0)>=1 and q.get('length_ok',0)>=1 and s>=.75
        and q.get('alignment_proxy',0)>=.65 and q.get('final_supported_by_reasoning',0)>=.55)


def read_filtered(path, limit):
    out=[]
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                r=json.loads(line)
                if good(r): out.append(r)
                if len(out)>=limit: break
    if len(out)<16: raise RuntimeError(f"Poucos exemplos filtrados: {len(out)}")
    return out


def encode(tok,row,max_len=256):
    bos=tok.token_to_id('<bos>') or 1; eos=tok.token_to_id('<eos>') or 2
    p=keep_edges(tok.encode(row['prompt']+'\n<think>\n').ids,96)
    t=keep_edges(tok.encode(row['think']).ids,112)
    f=keep_edges(tok.encode('\n<final>\n'+row['final']).ids,44)
    ids=[bos]+p+t+f+[eos]; segments=[-100]+[1]*len(p)+[2]*len(t)+[3]*len(f)+[4]
    ids=ids[:max_len+1]; segments=segments[:len(ids)]
    return torch.tensor(ids[:-1]),torch.tensor(ids[1:]),torch.tensor(segments[1:])


def batch(tok,rows):
    triples=[encode(tok,r) for r in rows]; width=max(len(x) for x,_,_ in triples); pad=tok.token_to_id('<pad>') or 0
    xs=[];ys=[];ss=[]
    for x,y,s in triples:
        n=width-len(x);xs.append(F.pad(x,(0,n),value=pad));ys.append(F.pad(y,(0,n),value=-100));ss.append(F.pad(s,(0,n),value=-100))
    return torch.stack(xs).long(),torch.stack(ys).long(),torch.stack(ss).long()


def load_model(path,vocab):
    m=PensamentoLinguistico(vocab_size=vocab,d_model=384,nhead=6,num_layers=4,dim_feedforward=1536,max_len=256)
    p=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(p.get('model_state_dict',p.get('model',p)),strict=False);return m


def forward(model,x): return model(x,torch.zeros((x.size(0),11),dtype=torch.float32))


def segment_scores(model,x,y,s,eos):
    model.eval(); flat=forward(model,x).flatten(0,1); fy=y.flatten(); fs=s.flatten(); out={}
    for key,tag in [('prompt',1),('think',2),('final',3),('eos',4)]:
        mask=(fs==tag)&(fy!=-100)
        out[key+'_tokens']=int(mask.sum())
        out[key+'_loss']=float(F.cross_entropy(flat[mask],fy[mask]).item()) if mask.any() else None
        out[key+'_accuracy']=float((flat.argmax(-1)[mask]==eos).float().mean().item()) if key=='eos' and mask.any() else None
    return out


def weighted_loss(model,x,y,s,eos,weights):
    flat=forward(model,x).flatten(0,1);fy=y.flatten();fs=s.flatten();valid=fy!=-100
    raw=F.cross_entropy(flat,fy,ignore_index=-100,reduction='none'); w=torch.ones_like(raw)
    for tag,val in weights.items(): w[fs==tag]=val
    return (raw[valid]*w[valid]).mean()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--rows',type=int,default=68);ap.add_argument('--steps',type=int,default=50);args=ap.parse_args()
    random.seed(20260829);torch.manual_seed(20260829);torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    rows=read_filtered(args.data,args.rows);Path(args.out/'filtered_dataset.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    tok=Tokenizer.from_file(str(args.tokenizer));cut=max(8,int(len(rows)*.8));train,val=rows[:cut],rows[cut:];model=load_model(args.base,tok.get_vocab_size());inject(model)
    for p in model.parameters():p.requires_grad=False
    trainable=[p for n,p in model.named_parameters() if 'lora_' in n]
    for p in trainable:p.requires_grad_(True)
    eos=tok.token_to_id('<eos>') or 2;xv,yv,sv=batch(tok,val);before=segment_scores(model,xv,yv,sv,eos)
    opt=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=.01);weights={1:.15,2:1.0,3:.5,4:4.0}
    for step in range(1,args.steps+1):
        start=((step-1)*4)%len(train);chosen=[train[(start+j)%len(train)] for j in range(4)];x,y,s=batch(tok,chosen);model.train();opt.zero_grad(set_to_none=True);loss=weighted_loss(model,x,y,s,eos,weights);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step()
        if step==1 or step==args.steps or step%25==0:print(f'step={step}/{args.steps} weighted_loss={loss.item():.6f}',flush=True)
    after=segment_scores(model,xv,yv,sv,eos);adapter={n:p.detach().cpu() for n,p in model.state_dict().items() if 'lora_' in n};result={'experiment':'pensamento_lora_multitarefa_continuidade','data':str(args.data),'base_checkpoint':str(args.base),'rows':len(rows),'train_rows':len(train),'val_rows':len(val),'steps':args.steps,'context_window':256,'loss_weights':{'prompt':.15,'think':1.0,'final':.5,'eos':4.0},'filter':{'score_min':.75,'alignment_min':.65,'final_support_min':.55,'structural_fields_required':True},'before':before,'after':after,'trainable_parameters':sum(p.numel() for p in trainable),'teacher_approved':False,'storage_root':'D:\\brothertrue','production_checkpoint_untouched':True}
    torch.save({'adapter_state_dict':adapter,'config':{'rank':8,'alpha':16,'context_window':256,'base_checkpoint':str(args.base),'targets':['decoder.linear1','decoder.linear2','lm_head']},'metrics':result},args.out/'adapter.pt')
    (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');journal=Path('D:/brothertrue/training_memory_journal.jsonl');journal.parent.mkdir(parents=True,exist_ok=True);with journal.open('a',encoding='utf-8') as fh:fh.write(json.dumps({'timestamp':result['experiment'],'result_path':str(args.out/'result.json'),'adapter_path':str(args.out/'adapter.pt'),'filtered_rows':len(rows),'production_checkpoint_untouched':True},ensure_ascii=False)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

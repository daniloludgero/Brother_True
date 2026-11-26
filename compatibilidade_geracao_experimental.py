"""Camada experimental de geração contextual para a variante Voz Chat Special."""
from __future__ import annotations
import re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from . import treinar_lora_voz, treinar_sft_chat_special

treinar_lora_voz.LoRALinear.weight = property(lambda self: self.base.weight)
treinar_lora_voz.LoRALinear.bias = property(lambda self: self.base.bias)

class GeradorContextualCompativel:
    def __init__(self, base: Path, tokenizer: Path, adapter: Path|None=None, max_len: int=128):
        self.tokenizer=Tokenizer.from_file(str(tokenizer)); self.max_len=max_len
        self.model=treinar_sft_chat_special.expanded_base(base,16384,self.tokenizer.get_vocab_size(),self.tokenizer,max_len)
        if adapter:
            treinar_lora_voz.add_lora(self.model,16,32,0.05)
            payload=torch.load(adapter,map_location='cpu',weights_only=False)
            self.model.load_state_dict(payload.get('adapter_state_dict',payload.get('adapter',{})),strict=False)
        self.model.eval(); self.bos=self.tokenizer.token_to_id('<bos>') or 1; self.eos=self.tokenizer.token_to_id('<eos>') or 2
        self.im_end=self.tokenizer.token_to_id('<|im_end|>'); self.end_text=self.tokenizer.token_to_id('<|end_of_text|>')

    def formatar(self, query: str, system: str='You are a helpful assistant. Answer clearly and concisely.') -> str:
        return f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

    @torch.no_grad()
    def responder(self, query: str, max_new: int=48) -> dict:
        prompt=self.formatar(query); prompt_ids=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        budget=max(1,self.max_len-max_new-1); prompt_ids=prompt_ids[-budget:]; ids=[self.bos]+prompt_ids; generated=[]; reason='max_new'
        x=torch.tensor([ids],dtype=torch.long); humor=torch.zeros((1,11),dtype=torch.float32)
        for _ in range(max_new):
            if x.size(1)>=self.max_len: reason='context_limit'; break
            nxt=int(self.model(x,humor)[0,-1].argmax()); generated.append(nxt); x=torch.cat([x,torch.tensor([[nxt]])],1)
            if nxt==self.eos: reason='eos'; break
            if nxt==self.im_end: reason='im_end'; break
            if nxt==self.end_text: reason='end_of_text'; break
        text=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        text=re.sub(r'<think>.*?</think>','',text,flags=re.I|re.S); text=re.sub(r'</?(?:think|final)>','',text,flags=re.I).strip()
        return {'prompt_tokens':len(prompt_ids),'generated_tokens':len(generated),'stop_reason':reason,'raw_text':self.tokenizer.decode(generated,skip_special_tokens=False),'public_text':text,'nonempty':bool(text),'special_ids':{'eos':self.eos,'im_end':self.im_end,'end_text':self.end_text}}

"""Adiciona marcadores de chat como tokens especiais sem alterar tokens antigos."""

from __future__ import annotations

import argparse
from pathlib import Path
from tokenizers import Tokenizer


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    tok=Tokenizer.from_file(str(a.source)); old=tok.get_vocab_size()
    added=tok.add_special_tokens(["<|im_start|>","<|im_end|>","<|end_of_text|>"])
    a.output.parent.mkdir(parents=True,exist_ok=True); tok.save(str(a.output))
    print({"old_vocab":old,"added":added,"new_vocab":tok.get_vocab_size(),"ids":{x:tok.token_to_id(x) for x in ["<|im_start|>","<|im_end|>","<|end_of_text|>"]}},flush=True)


if __name__=="__main__": main()

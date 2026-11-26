"""Tokenizer específico para prompts e traces privados do Pensamento."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def iterator(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = (item.get("prompt", "") + "\n<think>\n" + item.get("response", "")).strip()
            if text:
                yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    args = parser.parse_args()
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(r"\s+", " "), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=False)])
    tok.decoder = decoders.BPEDecoder(suffix="</w>")
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
                                  special_tokens=["<pad>","<bos>","<eos>","<unk>"],
                                  end_of_word_suffix="</w>", show_progress=True)
    tok.train_from_iterator(iterator(args.data), trainer=trainer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.output))
    print(json.dumps({"output":str(args.output),"vocab_size":tok.get_vocab_size()},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()

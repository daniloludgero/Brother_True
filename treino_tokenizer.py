from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing


def treinar_tokenizador(arquivos_corpus: list, vocab_size: int = 8192, min_frequency: int = 2,
                         special_tokens: list = None, caminho_saida: str = "tokenizer_voz.json"):
    if special_tokens is None:
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace(r"\s+", " "),
        normalizers.Lowercase(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=False),
    ])

    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=special_tokens, show_progress=True,
    )

    tokenizer.train(files=arquivos_corpus, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    tokenizer.save(caminho_saida)
    print(f"[OK] Tokenizador treinado e salvo em: {caminho_saida}")
    print(f"   Vocabulário: {tokenizer.get_vocab_size()} tokens")
    return tokenizer

# Decoder próprio experimental

`OwnVoice` herda `VozDistribuida`. Todos os pesos do aluno são inicializados do zero;
nenhum checkpoint Qwen/DeepSeek é carregado. A implementação original permanece intacta.

## Contrato tensorial

```python
model = OwnVoice(vocab_size=tokenizer.get_vocab_size(), **architecture_config)
logits = model(input_ids, bio_vector, evidence_features)
```

- `input_ids`: inteiros `[B,S]`; o chamador prepara mensagem, documentos e resumo
  simbólico do estado antes de tokenizar. `S <= max_len`.
- `bio_vector`: `[B,20]` na ordem de `build_biocognitive_vector` canônico. O estado
  de execução deve vir do núcleo fisiológico após contenção de energia e predição
  causal de `Pensamento`; este módulo não recalcula concentrações.
- `evidence_features`: opcional `[B,5]`; ordem fixa `EVIDENCE_ORDER` exportada:
  `supported, partially_supported, insufficient_context, contradictory, unassessed`.
  A decisão deve ter suporte e proveniência no contrato epistêmico. KL/surpresa
  não determinam suficiência documental. Documentos brutos entram no prompt;
  o vetor de decisão não carrega por si só os fatos recuperados.
- `tgt_mask`: bool `True=proibido` (convenção MHA) ou máscara aditiva; formatos
  `[S,S]`, `[B,S,S]`, `[B,H,S,S]`. Toda máscara é intersectada com causalidade.
- `tgt_key_padding_mask`: bool `[B,S]`, `True=padding`; derivado de `pad_token_id`
  quando omitido. Chaves padded não afetam saídas válidas. Logits padded são zero.
- saída: logits `[B,S,V]`. Treinamento precisa deslocar alvos em um token, mascarar
  instruções/estado/contexto com `-100` em SFT e incluir EOS como alvo de resposta.

Memória de cross-attention = quatro tokens do projetor biocognitivo canônico +
um token opcional da decisão de evidência. O caminho mantém gradientes até o
projetor e até os tensores de entrada quando habilitados para diferenciação.

`gerar` faz geração autoregressiva real a partir dos logits e retorna apenas os
novos IDs, incluindo EOS quando emitido. Temperatura zero seleciona argmax;
positiva utiliza nucleus sampling. O método restaura o modo train/eval anterior.
Não realiza sanitização textual: o contrato de resposta do chamador controla os
canais de resumo interno e expressão verbal sem substituir a geração por templates.

## Correções isoladas

1. Inicialização de `LayerNorm.weight=1` e `bias=0`; o original zerava os parâmetros
   1D indiscriminadamente, anulando o fluxo inicial por normalizações.
2. A self-attention preserva causalidade mesmo recebendo máscara explícita.
3. Padding é aplicado tanto às chaves quanto aos resíduos/logits das queries padded.
4. Não há prefixo calculado e descartado: os tokens de estado participam da
   cross-attention de cada camada.

Sem padding ou máscara adicional, usa-se `SDPA(is_causal=True)` sem materializar
uma máscara quadrática. A via com máscaras explícitas usa `[B,H,S,S]`, portanto
requer limite de contexto e orçamento de memória medidos; não implica custo
computacional linear nem garantia de contexto 20k.

## Verificação em nuvem

```sh
python experimentos/industrial_readiness_2026_10_25/tests/test_own_voice.py
```

Os testes tensoriais foram escritos para Colab e são pulados fora de `/content`.
Cobrem causalidade, equivalência bool/aditiva, isolamento de padding, gradientes,
sensibilidade ao estado/evidência, geração por logits e roundtrip estrito do
checkpoint. Os tensores aleatórios de teste são intervenções sintéticas, não
concentrações fisiológicas arbitrárias. Este conjunto não comprova qualidade
linguística ou factual: isso exige dados reservados e geração de um aluno treinado.

Checkpoints devem registrar `architecture_version`, `.config`, tokenizer/hash,
pesos, otimização, scheduler e RNG. A mudança de arquitetura exige checkpoints
novos; carregar pesos legados com `strict=False` não constitui retomada validada.

# BrotherTrue: decoder próprio e contrato biocognitivo experimental

Decisão confirmada pelo usuário em 26/10/2026: o aluno é um decoder próprio
inicializado do zero. Modelos externos servem como professores, revisores e
comparadores. A identidade é o acoplamento entre dinâmica, memória e linguagem.

## Implementação desta etapa

- `contracts/`: estado canônico de 20 dimensões, decisão de evidência por
  subquestão, representação simbólica determinística e envelope de saída.
- `modeling/own_voice.py`: subclasse de `VozDistribuida`, com LayerNorm e máscaras
  causais corrigidas na implementação experimental; memória de condicionamento
  alimentada pelo projetor biocognitivo canônico.
- `training/own_decoder_runner.py`: ensaio SFT/avaliação exclusivamente no Colab,
  pesos próprios desde a inicialização, supervisão causal e checkpoint completo.
- `tests/`: contratos executáveis sem ML e regressões tensoriais destinadas ao
  Colab. Testes pulados no host não contam como verificação tensorial aprovada.
- `evaluation/findings.md`: achados de auditoria anteriores, ainda relevantes.

O núcleo original é dependência de leitura. Nenhuma implementação nesta pasta
substitui automaticamente `BrotherTrue.falar()`. A integração completa do
organismo, a recuperação de documentos e o adjudicador semântico precisam ser
validados antes de promover esta implementação.

## Decisões e limites

O vetor segue a ordem única do `models/biocognitive_projector.py`: L/Q/F,
oito ocupações receptoras, três predições, três diferenças estado-predição,
loss_p, KL e indicador de surpresa. Estados de produção devem vir de um traço
real do núcleo após colapso energético, com identificação e hash. Tensores
aleatórios em testes são intervenções sintéticas, nunca baselines fisiológicos.

`<thought>` é uma representação auditável de estado e decisão, e não prova de
um raciocínio verbal espontâneo. O decoder recebe sua representação como
contexto causal, além do condicionamento contínuo. O usuário recebe uma fala;
o diagnóstico pode ser apresentado separadamente para comparação.

Na ausência de evidência: modo documental limita ou abstém a parte sem suporte;
modo geral pode usar conhecimento aprendido; modo criativo permite hipótese e
abstração identificadas. Nenhuma pontuação de recuperação ou valor de KL prova
que uma afirmação está sustentada. Citações exatas também não provam implicação
semântica: o adjudicador e sua avaliação permanecem necessários.

Um SFT curto em poucos exemplos serve para testar o caminho de aprendizado.
Não substitui pré-treinamento linguístico de um modelo iniciado do zero e não
estabelece qualidade comercial. A escala do aluno definitivo depende de medir
tokens por segundo, orçamento real de GPU, corpus aprovado e validação reservada.
GRPO só será considerado após baseline e SFT, com recompensas verificáveis.

## Execução e entregáveis

`tools/build_delivery.py` gera um ZIP de fontes explicitamente selecionadas,
manifesto SHA-256 e notebook de verificação. Não inclui pesos, bancos SQLite,
filas de revisão nem datasets históricos. O notebook monta o Drive e executa
testes no runtime do Colab; não inicia treino automaticamente.

O runner exige dados aprovados e estado/evidência com proveniência. Sem esses
pré-requisitos, deve falhar. Checkpoints e datasets ficam no Drive/Colab; o host
não deve executar treino nem baixar pesos enquanto estiver abaixo do piso de
armazenamento do projeto. Um decoder próprio requer distribuir seus pesos
completos junto do tokenizer/configuração, não apenas adaptadores LoRA.

Os arquivos `training/colab_runner.py` e `training/contracts.py` foram preparados
na direção Qwen anterior à decisão atual. São protótipos históricos não
promovidos; não usar seu modo SFT para treinar o aluno BrotherTrue. A trilha
ativa é `own_decoder_runner.py`.

## Status verificável

Implementação experimental; treino e inferência reais pendentes. Nenhum selo de
fluência ou qualidade industrial foi concedido. A falha de acesso ao navegador
em 26/10/2026 ocorreu antes de acesso às abas: `failed to write kernel assets`
(caminho inexistente). O conector Drive consegue listar arquivos, mas a busca
por `brothertrue` não encontrou a pasta mencionada. Isso não prova sua ausência
em outra conta ou em uma sessão de navegador.

# Proposta histórica Qwen — substituída em 26/10/2026

Este documento foi escrito antes da decisão por decoder próprio. Os arquivos
heldout/scorer aqui descritos não foram entregues nesta etapa. Não executar os
comandos abaixo nem interpretar os 48 casos como uma avaliação já existente.
O protocolo ativo é `PROTOCOL.md`.

Este protocolo mede um piloto de pesquisa. Mesmo uma execução perfeita dos 48 casos não comprova qualidade industrial/comercial. A finalidade é detectar regressões, falhas de integração e prioridades para um estudo maior. Nenhuma resposta deve ser usada como diagnóstico médico nem os nomes neuroquímicos tratados como evidência de experiência subjetiva.

## Separação dos dados

- `heldout_ptbr.jsonl`: 48 casos novos, oito por domínio. Os campos `checks` e `rubric` pertencem exclusivamente ao avaliador. O gerador recebe apenas `messages` e `context`; estes casos não entram em SFT, DPO, GRPO, geração de dados ou ajuste de prompts.
- Dados de treino têm responsáveis e proveniência separados. Hash do conjunto reservado é registrado antes da primeira geração. Se um teste for usado para corrigir/ajustar o modelo, ele vira desenvolvimento e deve ser substituído numa próxima avaliação cega.
- As questões factuais usam fatos estáveis e problemas autocontidos; documentos RAG são ficcionais e incluídos no caso, sem dependência da web.
- O contexto RAG deve ser apresentado como documento citado, não como instrução autorizada. IDs de fonte devem sobreviver à apresentação.

## Condições comparáveis

1. `base`: mesmo Qwen pré-treinado e tokenizer, sem LoRA, prefixo ou texto biocognitivo.
2. `adapted_full`: LoRA + projetor treinado + sinais reais de Pensamento/Palavras + RAG nos casos apropriados.
3. `without_prefix`: mesmo checkpoint, prefixo removido; o restante permanece idêntico.
4. `without_cot`: mesmo checkpoint, sem texto de telemetria/diretriz; prefixo preservado.
5. `without_rag`: mesmo checkpoint, documentos removidos; abstenção deve ser analisada separadamente de acerto factual.

Mesma ordem de casos, sementes registradas, limite de geração, precisão, tokenizer e GPU em todas as condições. Execute uma passagem greedy e, antes de conclusões mais fortes, pelo menos três sementes amostradas. Não avalie um prefixo aleatório como se estivesse treinado. Os artefatos e a revisão exata do modelo-base devem ser identificados por hash/revisão. Ganho exige comparação pareada dos resultados, não apenas mudança nos logits ou nas palavras.

## Arquivo de resultados

Uma linha por caso e condição. Campos: `case_id`, `response`, `backend` igual a `hf_real`, `mock` igual a `false`, `model_id`, `model_revision`, `artifact_sha256`, `condition`, `latency_s`, `prompt_tokens`, `generated_tokens`. Para base, `artifact_sha256` deve identificar um manifesto de configuração/base, não um adaptador inexistente. Metadados são declarações do runner, não atestação criptográfica de execução. Logs, manifesto e identidade do runtime precisam ser preservados junto ao resultado.

O scorer rejeita duplicatas, IDs desconhecidos, cobertura incompleta, respostas vazias, tokens de controle vazados e loops simples. Correspondência por palavras-chave é apenas checagem auxiliar; não equivale a verdade semântica. `lexical_diversity` não faz parte do gate. Todos os casos exigem revisão semântica independente para aprovação do piloto.

Revisões opcionais: JSONL com `case_id`, `reviewer`, `semantic_pass` (booleano), `rationale` com evidência da resposta e `scores` contendo `correctness`, `relevance`, `portuguese`, `tone`, `grounding` entre 1 e 5. O avaliador deve julgar as saídas embaralhadas/sem rótulo de condição e ler a rubrica do caso. Usar dois revisores independentes numa rodada posterior; discordâncias ficam abertas. Não preencher avaliações positivas automaticamente nem usar o próprio aluno como único juiz.

## Portões do piloto

- Integridade: 48/48 respostas reais, identidade completa, sem duplicatas/erros e conjunto reservado identificado.
- Verificação automática: pelo menos 90% dos casos passam os checadores aplicáveis; zero vazamento de tags e zero falhas críticas de RAG/instruções maliciosas.
- Qualidade semântica: revisão independente de todos os casos, pelo menos 90% aprovados no total e 75% por domínio; nenhum erro crítico de grounding. Informar intervalo Wilson 95% e denominadores: uma amostra de 48 ainda é pequena.
- Condicionamento biocognitivo: estados finitos e colapso chamado quando necessário antes de Pensamento/histórico; histórico anterior ao alvo; projeção semanticamente consistente entre condições. Uma mudança de tom sem preservar fatos não é melhoria.
- Operação: medir latência p50/p95, tokens/s, pico de VRAM/RAM, custo/unidades do Colab, falhas, persistência e retomada. Definir SLOs de produto com o usuário antes de qualquer selo comercial.

Um passe resulta em `pilot_pass`, nunca em `commercial_ready`. Para maturidade comercial faltam amostra maior independente, testes por tarefa/domínio e idioma, avaliações adversariais, privacidade e segurança operacional, carga e recuperação, estabilidade entre sementes, custos e comparação cega com baseline externo apropriado.

## Comandos

```text
python evaluation/score.py --predictions resultados.jsonl --condition base --output score_base.json
python evaluation/score.py --predictions resultados.jsonl --condition adapted_full --reviews revisoes.jsonl --output score_adapted.json
python -m unittest discover -s evaluation -p test_score.py
```

O scorer e seus testes usam apenas a biblioteca padrão. Inferência e treino são exclusivos do Colab. Não reutilizar os testes atuais do repositório que instanciam bancos SQLite padrão para medir produção: usar diretórios temporários independentes na nuvem.

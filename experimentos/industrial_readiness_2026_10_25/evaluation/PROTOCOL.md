# Avaliação do decoder próprio — protocolo ativo

Esta etapa entrega testes de contratos e regressões do decoder. Ainda não existe
conjunto de 48 casos reservado nem scorer semântico implementado. Esses itens
citados no rascunho Qwen histórico permanecem planejamento.

## Comparações necessárias

Use o mesmo tokenizer, checkpoint, mensagens, fontes e sementes:

1. Modelo com inicialização aleatória versus checkpoint treinado: demonstra
   aprendizado medido, sem presumir competência geral.
2. Modelo completo versus retirada dos tokens contínuos de biocognição.
3. Modelo completo versus retirada do resumo simbólico.
4. Modelo completo versus troca do estado por outro traço válido do núcleo.
5. Fontes suficientes versus fontes parciais, ausentes e conflitantes.
6. Modo documental versus geral e criativo, verificando identificação de hipótese.

O runner piloto implementa `full`, `without_bio` e `without_evidence` como
intervenções apenas no caminho da cross-attention. O resumo simbólico e os
documentos continuam no prompt. Isso não constitui ablação completa de
biocognição ou RAG. Os próximos braços precisam remover os respectivos
caminhos em conjunto e registrar exatamente o que foi alterado.

Logits diferentes e gradientes não nulos provam conexão no grafo computacional.
Ganho de qualidade exige julgamento sobre respostas. Não usar lexical diversity,
EOS ou loss isoladamente como selo de fluência.

## Conjunto reservado a produzir antes do treino de linguagem

Responsável independente do gerador de treino cria mensagens em PT-BR para
instruções, raciocínio verificável, conversa casual, humor/provocação respeitosa,
contexto multi-turno e RAG positivo/negativo/conflitante. Referências e rubricas
ficam fora da entrada do aluno e dos professores que criam dados de treino.

Congelar hashes antes da primeira geração. Casos usados para ajustar o sistema
passam a desenvolvimento e devem ser substituídos na próxima rodada cega.
Julgamentos docentes são evidências registradas, não aprovação automática.

## Relatório exigido

Registrar por caso: ID, resposta, backend próprio real (`mock=false`), hashes de
checkpoint/tokenizer/fontes, condição, sementes, tokens, latência, EOS e falhas.
Preservar saída antes do filtro em artefato diagnóstico de acesso controlado e
publicar somente a resposta limpa. O runner atual registra a resposta limpa;
persistência da saída bruta e revisão semântica ainda são melhorias pendentes.

Avaliar correção, pertinência, português, tom, memória contextual, grounding,
abstração identificada e calibração; revisar condições embaralhadas, idealmente
com dois avaliadores independentes. Medir latência p50/p95, tokens/s, VRAM/RAM,
unidades de computação, falhas e retomada. Reportar denominadores e incerteza.

Um piloto pode detectar limites e regressões; nenhuma execução deste protocolo
estabelece sozinha qualidade industrial. Critérios de produto, corpus linguístico
amplo, validação externa e testes operacionais continuam necessários.

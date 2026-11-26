# Contrato experimental do decoder próprio

Este pacote usa somente a biblioteca padrão. O aluno é o decoder próprio do
BrotherTrue, inicializado do zero; professores externos podem revisar dados,
ensinar por destilação e avaliar. A representação `<thought>` aqui é uma projeção
determinística auditável, não uma alegação de raciocínio verbal gerado pelo LLM.

## Interface

`BiocognitiveState.from_core(vector20, source=..., energy_crit=...,
collapse_checked=True)` recebe um snapshot destacado do núcleo, depois de
`apply_collapse`, `Pensamento` e `Palavras`. O layout é o canônico de
`models.biocognitive_projector.build_biocognitive_vector(..., include_delta=True)`:
X3, oito ocupações de receptor, predição3, diferença3, loss_p, KL, surpresa.
Não há preenchimento de receptor com valores inventados. `source` e o indicador
de colapso são declarações de proveniência do chamador: não autenticam o núcleo.
O chamador deve registrar fonte, versão, turno e ordem causal. `energy_crit` vem
de `ode.E_crit(age)` na implementação atual; o significado de `age` deve ser
preservado até uma revisão científica explícita. Energia ainda acima desse
limiar é rejeitada mesmo se o chamador declarou ter avaliado colapso.

`EvidenceSource(id,text)` carrega um documento como dado não confiável.
`Citation(source_id,quote)` exige trecho literal existente na fonte.
`EvidenceClaim(id,subquestion,verdict,citations,adjudicator,adjudication_id)`
representa uma decisão externa explícita por subquestão. Um veredito favorável,
parcial ou contraditório exige citações e proveniência de revisão. Este contrato
confere forma e vínculo textual; não prova que a citação implica a resposta ou
que o nome do revisor é autêntico. O integrador deve fornecer decisões de um
adjudicador confiável e manter o comprovante completo de revisão. Nenhum score
de busca, proximidade lexical ou presença de documento promove suporte factual.

`EvidenceDecision(mode,claims,sources)` agrega decisões conservadoramente:
qualquer conflito permanece conflito; partes sustentadas + lacunas tornam o
conjunto parcial; ausência de revisão fica insuficiente. Os modos resolvem a
distinção entre abstração e abstenção discutida com o usuário:

- `grounded`: responder apenas o sustentado, explicitar lacunas e conflitos;
  uma hipótese pode ser proposta se identificada como tal, nunca como fato da fonte.
- `general`: permite conhecimento paramétrico e abstração sem RAG, mantendo
  explícita a diferença entre inferência e suporte documental.
- `creative`: permite ficção/hipóteses identificadas, sem atribuição factual falsa.

`ThoughtRenderer.render(state,decision)` gera o bloco de condicionamento com
números, modos, vereditos, IDs e diretivas fixas. Texto recuperado, pergunta e
citações não entram nesse canal de controle. Precisam ser entregues ao decoder
em um campo delimitado de **dados**, com proveniência e defesa contra injeção.
O resumo do renderer pode ser exibido opcionalmente como diagnóstico comparativo.

`build_response(raw,state,decision)` entrega `ResponseEnvelope.response` como
única fala e `.diagnostic` como canal separado. Tags internas balanceadas são
removidas; tags truncadas, malformadas ou controles de papel na fala bloqueiam
a publicação inteira. É necessário armazenar a saída antes de publicar: não
transmitir tokens crus em streaming. Essa defesa de canal não prova verdade,
fluência, atendimento à diretiva ou ausência de telemetria em linguagem natural.

## Verificação científica antes de qualquer selo de qualidade

Manter perguntas, documentos, respostas de referência e avaliações reservadas
fora do corpus de treino. Executar o mesmo conjunto e sampler por checkpoint em
quatro condições: estado/evidência corretos; estado removido; estado trocado entre
casos; evidência removida ou contraditória. Usar também ablações independentes
do prefixo contínuo e do bloco simbólico para detectar atalhos entre caminhos.

Medir diferenças de logits **e** mudança apropriada na resposta: uma diferença
numérica só demonstra dependência, não melhora de comportamento. Para RAG,
medir suporte por subquestão, precisão das citações, falsas alegações e abstenção
calibrada; avaliar separadamente `grounded`, `general` e `creative`. O aumento de
loss_p deve poder mudar a cautela sem destruir correção factual; receptores não
devem mudar fatos verificáveis. Avaliadores independentes precisam julgar
fluência e casualidade em PT-BR, resposta à instrução, continuidade multi-turno,
provocação respeitosa, limites de conhecimento e resistência à injeção.

Um treino que reduz loss ou aumenta diversidade lexical não comprova fluência
nem qualidade industrial. Pilotos pequenos validam plumbing e identificam
falhas; conclusões comerciais exigem amostra representativa, intervalos de
incerteza, custos/latência/recursos medidos e reprodução por artefato identificado.
Registrar o tokenizer e seu SHA-256, configuração do decoder, inicialização,
versão destes contratos, checkpoints/RNG, splits e todos os julgamentos completos.

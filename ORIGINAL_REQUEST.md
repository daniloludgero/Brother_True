# Original User Request

## 2026-10-18T21:18:53Z

Estabelecer um sandbox isolado de experimentação para auditar rigorosamente a capacidade generativa do modelo de Voz do BrotherTrue (partindo do checkpoint checkpoints_voz_lora_sft/best.pt), diagnosticar inconstâncias verbais/semânticas e atuar como Modelo Professor (Gemini) em nível doutorado, combinando a curadoria da review_queue existente com a síntese de novos dados de alta coerência, evoluindo o treinamento da Voz sem alterar o código-fonte do modelo original.

Working directory: /mnt/c/Users/Samsung/Downloads/brothertrue/sandbox_eval
Integrity mode: benchmark

## Requirements

### R1. Sandbox Isolado de Avaliação Generativa
Criar um ambiente de testes isolado dentro de sandbox_eval/ com scripts dedicados para auditar a capacidade geradora da Voz sem alterar qualquer arquivo do núcleo do projeto. O sandbox deve calcular métricas objetivas: diversidade léxica (unique_ratio_words), taxa de terminação correta por <eos>, perplexidade/loss de validação e estabilidade ante variações do vetor de humor de 11 dimensões.

### R2. Diagnóstico Profundo das Inconstâncias Verbais e Semânticas
Conduzir avaliação sistemática do modelo baseline (checkpoints_voz_lora_sft/best.pt) mapeando as falhas conhecidas (loops de repetição de caracteres/aspas, truncamento antecipado e descolamento semântico), gerando um relatório canônico de diagnóstico pré-treino com evidências quantitativas e amostras de geração.

### R3. Síntese e Curadoria Híbrida da Base pelo Professor (Gemini)
Como modelo orientador e técnico de nível doutorado:
1. Auditar e filtrar a fila de revisão (review_queue) de professores prévios (DeepSeek e Qwen), descartando pares ruidosos ou degenerados.
2. Sintetizar novos pares instrucionais e conversacionais de alta fidelidade e riqueza léxica, com marcação estrita de tokens especiais e terminação formal em <eos>.
3. Catalogar o dataset consolidado resultante com manifest estruturado, contendo contagem de tokens, schema e hash SHA-256 de proveniência.

### R4. Ciclo de Treinamento Adaptativo Supervisionado (SFT)
Executar o treinamento supervisionado no sandbox sobre o novo dataset curado, mantendo o código-fonte do modelo intocado. O processo de treino deve obrigatoriamente operar sob o ResourceGovernor (limites cooperativos de CPU/RAM) e utilizar checkpoint_manager_safe para salvamento atômico dos pesos e dos estados de geradores pseudoaleatórios (Python, NumPy, PyTorch RNG).

### R5. Validação Comparativa e Homologação Técnica
Executar bateria de testes idêntica pós-treinamento no sandbox, comparando diretamente as métricas antes vs. depois. Produzir relatório técnico final corporativo detalhando os ganhos obtidos, limitações remanescentes e recomendações para próximas iterações.

## Acceptance Criteria

### Isolamento Estrito do Código-Fonte
- [ ] Nenhum arquivo do núcleo do projeto (brothertrue.py, ode.py, neuroquimica.py, memoria.py, pensamento.py, palavras.py, ciclo_rem.py, voz.py) é modificado. Todas as adições ocorrem estritamente dentro de sandbox_eval/.

### Diagnóstico Baseline Objetivo
- [ ] Relatório pré-treino gerado com amostragem estatística de geração, registrando formalmente unique_ratio_words, taxa de encerramento por <eos> e loss de validação.

### Base de Dados Curada e Rastreável
- [ ] Dataset consolidado em formato JSONL contendo pares de treino validados e documentados em sandbox_eval/dados/manifest.json com contagem de linhas e hash SHA-256.

### Estabilidade do Treinamento e Checkpoints
- [ ] Treinamento executado sob ResourceGovernor.throttle() com threads de CPU restritas (torch.set_num_threads(2) ou 4).
- [ ] Checkpoint final salvo contendo pesos, otimizador e estados completos de RNG para reprodutibilidade.

### Validação de Resultados e Redução de Inconstâncias
- [ ] O relatório comparativo final atesta aumento quantitativo na diversidade léxica (unique_ratio_words >= 0.55) e eliminação de loops de repetição de caracteres únicos/aspas presentes no diagnóstico histórico.
- [ ] Parecer de Aval Técnico emitido com análise de causa-raiz e direcionamento para os próximos passos.

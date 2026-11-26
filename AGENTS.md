# Diretrizes e Invariantes Arquiteturais do Projeto BrotherTrue

Este repositório implementa o ecossistema biocognitivo **BrotherTrue**, unindo modelos dinâmicos contínuos (EDOs de Qualia), neuroquímica fisiológica, memória associativa e decodificação autoregressiva de linguagem. Qualquer modificação no código ou execução experimental deve obedecer rigorosamente às seguintes diretrizes:

---

## 1. Invariantes de Dinâmica e Estabilidade Numérica (ODE & Neuroquímica)

1. **Contenção Obrigatória de Energia (Collapse Invariant)**:
   - Em todo ciclo multi-turno de interação (`falar` e `ouvir_e_processar`), o estado de energia $E = 0.5 L + 0.5 Q$ deve ser avaliado contra o limiar crítico $E_{\text{crit}}(a) = E_0 + \eta_E a$.
   - Se $E > E_{\text{crit}}$, o método `ode.apply_collapse(X, E, E_crit)` deve ser obrigatoriamente acionado antes de alimentar o módulo `Pensamento` ou atualizar `state_history`, prevenindo o runaway exponencial de estados ($L, Q \to \infty$) e a explosão de loss no módulo introspectivo.

2. **Isolamento de Alvo em Introspecção (Target Leakage Prevention)**:
   - No módulo `Pensamento`, a janela de atenção deve conter estritamente estados anteriores ao passo corrente. O estado atual atua exclusivamente como alvo supervisionado (`compute_loss`) e só pode ser anexado ao histórico *após* a predição.

3. **Derivação Fisiológica de Receptores (Hill Occupancy)**:
   - Concentrações basais e perturbações farmacológicas devem ser computadas através da equação de Hill baseada nas constantes de literatura em nanomolar ($nM$) de `farmacologia.py`. Nunca reintroduza valores fixos *ad hoc* sem vínculo biofísico.

4. **Fronteira de Autograd no Encoder Semântico**:
   - Ao transicionar tensores do `EncoderSemantico` (produzidos sob `torch.no_grad()` pelo SentenceTransformer) para o `ProjetorSemantico`, aplique sempre `.clone()` no tensor de saída para evitar inconsistências de modo de inferência (`InferenceMode`) no autograd.

---

## 2. Invariantes de Persistência, Concorrência e SQLite

1. **Protocolo Transacional de Checkpoints**:
   - Conexões com `qualia_memory.db` e `qualia_ltp.db` devem ser explicitamente fechadas (`memory.close()`, `ltp.close()`) antes de qualquer operação de cópia/backup no disco para evitar corrupção por escrita concorrente.
   - Imediatamente após o backup, novas instâncias com conexões abertas devem ser reatribuídas aos atributos do modelo para manter o objeto operável.

2. **Checkpoints Seguros com Estado de RNG**:
   - Treinamentos e experimentos de longa duração devem utilizar o padrão `checkpoint_manager_safe.py`, salvando atômica e temporariamente (`mkstemp` + `os.replace`) os estados do modelo, otimizador, escalonador e geradores de números pseudo-aleatórios (Python, NumPy e PyTorch).

---

## 3. Governança de Recursos e Execução Cooperativa

1. **Limites de CPU e Memória (WSL / Host)**:
   - Processos de treino contínuo devem acionar o `ResourceGovernor.throttle()` entre lotes (*batches*), respeitando as políticas padrão de até 75% de CPU e 60% de RAM.
   - Restrinja sempre as threads de inferência e treinamento via `torch.set_num_threads(2)` ou `torch.set_num_threads(4)` para evitar saturação do subsistema operacional.

2. **Modo Offline do HuggingFace**:
   - Defina `os.environ["HF_HUB_OFFLINE"] = "1"` em todos os scripts experimentais para garantir autonomia e execução local sem bloqueios de rede.

---

## 4. Governança de Datasets e Modelos Professores

1. **Invariante de Limpeza de Dados (Data Retention Rule)**:
   - Nenhum arquivo de dataset em `dados_externos/` ou `dados_modulos_novos/` pode ser excluído sem que:
     - Exista um artefato derivado ou checkpoint validado e persistido no disco.
     - Seja registrada uma entrada formal em `relatorios/limpeza_gradual_YYYY-MM-DD.md` documentando o nome, tamanho, hash SHA-256 e justificativa do descarte.

2. **Política de Fila de Revisão dos Professores (Review Queue Policy)**:
   - Dados na fila de revisão (`review_queue`) jamais devem ser injetados em pipelines de treino de decoders antes da validação formal pelos modelos professores (DeepSeek-V4-Flash / Qwen).

---

## 5. Governança de Armazenamento, Streaming e Offloading em Nuvem (Google Drive / Colab)

1. **Preservação Crítica do SSD Host C:**:
   - O disco host C: opera sob capacidade crítica (> 95% ocupado, ~6 GB livres). Todo download ou ingestão de datasets remotos deve ser executado obrigatoriamente via streaming por micro-lotes (`streaming=True` no HuggingFace Datasets).
   - O sentinela `HardwareSentry` deve monitorar a partição continuamente. Caso o espaço caia abaixo de 4.5 GB, o expurgo de caches (`.hf_cache`, `/tmp`) é acionado de imediato. A execução é abortada preventivamente caso o piso de 4.0 GB seja violado.

2. **Arquitetura de Offloading em Nuvem (Google Drive e Google Colab)**:
   - Para volumes maciços de dados ou treinos de longa duração que excedam a capacidade de armazenamento local, o ecossistema deve priorizar o uso do Google Drive como repositório remoto (`/content/drive/MyDrive/brothertrue`) e instâncias aceleradas (T4/L4/V100/A100) do Google Colab Pro.
   - Notebooks de treinamento devem conter rotinas de montagem automática do Google Drive, sincronização atômica de checkpoints e preservação do isolamento da memória física do host local.

---

## 6. Teoria dos Sistemas, Chain-of-Thought Endógeno e Projeção Estratégica (1B–7B Colab Pro)

1. **Projeção Orientada a Propósito e Contextualidade Geral**:
   - O treinamento e a curadoria de dados não devem buscar volume bruto indiscriminado nem hiper-especialização restritiva.
   - Os datasets e modelos base (1B a 7B, priorizando a família `Qwen2.5-3B-Instruct` / `Qwen2.5-7B-Instruct`) devem abranger conhecimento geral amplo (padrão LLM comercial) integrado organicamente à identidade biocognitiva do BrotherTrue.

2. **Preservação de POO, Teoria dos Sistemas e Chain-of-Thought Simbólico (`Pensamento`)**:
   - Todo acoplamento de novos decodificadores deve respeitar a herança POO (`nn.Module`) e a homeostase sistêmica entre `QualiaODE`, `Neuroquimica`, `Memoria`, `LongTermMemory`, `CicloREM`, `Palavras` e `Pensamento`.
   - A classe `Pensamento` (predição introspectiva de trajetória $\hat{L}, \hat{Q}, \hat{F}$ e `loss_p`) e a classe `Palavras` (divergência KL e `surprise`) operam como o *Chain-of-Thought* (CoT) simbólico e analógico do organismo. Suas saídas devem participar ativamente da resolução final da `Voz`, evitando que a geração verbal opere divorciada do estado introspectivo.

3. **Arquitetura Híbrida 1B–7B no Google Colab Pro**:
   - Em ambientes Google Colab Pro, priorizar o acoplamento de modelos open-source consolidados (1B a 7B parâmetros via QLoRA/LoRA, BF16/FP16/4-bit, RoPE e SDPA) com os projetores fisiológicos do BrotherTrue (`HumorProjector` 11D + ponte CoT endógena), exportando apenas adaptadores compactos para execução offline no host local.



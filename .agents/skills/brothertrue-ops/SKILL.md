---
name: brothertrue-ops
description: >-
  Runbooks operacionais e procedimentos de teste, diagnóstico e ciclo de vida
  do ecossistema biocognitivo BrotherTrue. Use esta skill para executar o
  teste de Turing integrado (VT-1), diagnosticar explosão de estados ODE,
  conter colapso de saída do decoder Voz e gerenciar checkpoints e dados.
---

# BrotherTrue Operational & Engineering Runbooks

Este guia consolidado contém os procedimentos formais para operar, treinar e depurar o sistema biocognitivo BrotherTrue.

---

## 1. Execução do Teste de Turing Integrado (Protocolo VT-1)

O teste VT-1 valida a interoperabilidade ponta a ponta entre todos os subsistemas (`EncoderSemantico`, `ProjetorSemantico`, `QualiaODE`, `NeuroTransmissor`, `Pensamento`, `Palavras`, `Memory`, `LongTermMemory` e `Voz`).

### Procedimento de Execução

Execute o script de teste integrado com os argumentos canônicos:

```bash
python3 experimento_turing_brothertrue_integrado_v2.py \
  --base checkpoints_voz_lora_sft/best.pt \
  --tokenizer tokenizer_voz_novo_16k.json \
  --report relatorios/experimento_turing_homologacao.json \
  --turns 4 \
  --max-new 16
```

### Critérios de Aceitação e Validação
1. **Estabilidade de Estado**: Verifique no relatório JSON gerado se os valores de $L$, $Q$ e $E$ permanecem dentro da faixa estável ($< 50.0$) e não escalam exponencialmente.
2. **Loss do Pensamento**: O `loss_pensamento` deve manter-se abaixo de $10.0$ em todos os turnos.
3. **Diversidade Léxica da Voz**: O índice `unique_ratio_words` deve ser superior a $0.50$ (não repetitivo).

---

## 2. Diagnóstico e Correção de Anomalias Frequentes

### Anomalia A: Explosão de Estados no Ciclo Multi-Turno
- **Sintoma**: $L, Q, E$ dobram ou escalam por ordens de grandeza a cada chamada consecutiva a `falar()`, e o loss do `Pensamento` atinge valores exorbitantes ($> 10^5$).
- **Correção no Código**: Assegure que a chamada `ode.apply_collapse` é realizada após a integração numérica:

```python
# Trecho de contenção obrigatória em brothertrue.py
final_X = solucao[-1]
L, Q, F = final_X[0], final_X[1], final_X[2]
E = 0.5 * L + 0.5 * Q

# Contenção de colapso energético:
E_crit = self.ode.E_crit(self.age)
if E > E_crit:
    novo_X, colapsou = self.ode.apply_collapse(final_X, E, E_crit)
    L, Q, F = novo_X[0], novo_X[1], novo_X[2]
    E = 0.5 * L + 0.5 * Q
    self.neuro.atualizar(E.item(), L.item(), Q.item(), collapsed=True)
else:
    self.neuro.atualizar(E.item(), L.item(), Q.item(), collapsed=False)
```

### Anomalia B: Colapso Autoregressivo da Voz (Aspas ou Tokens Repetidos)
- **Sintoma**: Geração presa em tokens redundantes (`unique_ratio < 0.05`), conforme observado em `diagnostico_colapso_saida_voz.json`.
- **Procedimento de Recuperação**:
  1. Inspecione o alinhamento entre o vocabulário do tokenizer e `lm_head.out_features` via `testar_compatibilidade_geracao.py`.
  2. Ajuste o sampler com `temperatura=0.85`, `top_p=0.90` e penalidade de repetição.
  3. Verifique se o token `<eos>` está devidamente posicionado nos alvos de SFT.

---

## 3. Gestão de Recursos e Treino Seguro

Para qualquer rotina de fine-tuning ou destilação:

1. **Instanciar o Governador de Recursos**:
```python
from resource_governor import ResourceGovernor, ResourcePolicy

gov = ResourceGovernor(ResourcePolicy(cpu_limit_percent=70.0, ram_limit_percent=55.0))
```

2. **Chamar o throttle no loop de épocas**:
```python
for batch in dataloader:
    # ... treino do batch ...
    status = gov.throttle()
    if status["paused"]:
        print("[AVISO] Treinamento pausado temporariamente para alívio térmico/memória.")
```

3. **Salvar Checkpoint com Estado de RNG**:
```python
from checkpoint_manager_safe import save_checkpoint

save_checkpoint(
    "checkpoints/etapa_1.pt",
    model=modelo,
    optimizer=otimizador,
    scheduler=escalonador,
    epoch=epoca,
    step=passo,
    metrics={"loss": val_loss}
)
```

---

## 4. Consulta e Manutenção de Bancos SQLite

Para inspecionar ou auditar as memórias episódicas e de longo prazo:

```bash
# Total de memórias registradas por rótulo emocional
sqlite3 qualia_memory.db "SELECT label, COUNT(*), AVG(score) FROM memories GROUP BY label;"

# Memórias consolidadas de maior prioridade na LTP
sqlite3 qualia_ltp.db "SELECT id, label, consolidation_score, replay_priority FROM ltp ORDER BY replay_priority DESC LIMIT 10;"
```

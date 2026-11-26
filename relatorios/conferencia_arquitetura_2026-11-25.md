# Conferência de Arquitetura do Pipeline `falar` — 2026-11-25

**Status: AVAL CONDICIONAL** (ressalva de infraestrutura, não de código)

## Pipeline conferido (`BrotherTrue.falar`)
Mensagem → Encoder semântico → QualiaODE + Neuroquímica + Memória → Pensamento + Palavras → Registro biocognitivo (`BiocognitiveReasoning`, criado antes do decoder) → Evidence Gate (`models/evidence_gate.py`) → Renderer `<thought>` (`models/thought_renderer.py`) → Decoder `Voz` condicionado por `prompt_ids` → `DeepSeekCoTParser` (resposta limpa).

## Alterações
| Arquivo | Mudança |
|---|---|
| `models/evidence_gate.py` | novo: filtra evidências do banco vetorial e da LTP (grounded, confiança, margem, sanitização de tags) |
| `models/thought_renderer.py` | novo: renderiza o bloco `<thought>` a partir do registro e das evidências aprovadas |
| `voz.py` | `gerar(..., prompt_ids=None)`: prefixo opcional, truncado ao orçamento de `max_len`; padrão idêntico ao original |
| `brothertrue.py` | reordenação dos nós em `falar`; `_tokenizar`; atributos `evidence_gate`, `ultimo_gate`, `thought_no_decoder` |
| `tests/conferir_arquitetura.py` | novo: verificador estático (ordem das chamadas e herança), somente leitura |

Backup dos originais: `brain/<conversa>/scratch/backup/` (`brothertrue.py`, `voz.py`).

## Evidências
- `tests/conferir_arquitetura.py`: **CONFORME** (8/8 nós em ordem, herança OK).
- `py_compile` dos 4 arquivos alterados ou criados: OK.
- `unittest` (zero-leakage, biocognitive_bridge, e2e_ecosystem): 49 testes, **48 OK, 1 erro**.

## Ressalva
`test_resource_governance_invariants` falhou com `check_host_disk_safety`: **0,26 GB livres** em C: (mínimo 4,0 GB). É o guarda de preservação do SSD funcionando; a causa é o disco host, não a mudança. Não atribuo o erro ao código, mas o teste só será considerado verde depois de liberar espaço.

## Pendências (CAPA)
1. Liberar espaço em C: (expurgo de caches conforme a política da `HardwareSentry`; nenhum dataset deve ser apagado sem o registro de limpeza) e rodar o teste de novo.
2. Não há teste dedicado para o `EvidenceGate`, o `prompt_ids` da `Voz` nem a ordem dos nós. Recomendado antes de levar ao repositório.

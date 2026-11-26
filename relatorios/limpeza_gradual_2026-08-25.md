# Limpeza gradual de datasets

Primeira etapa: somente arquivos já reaproveitados ou pilotos vazios.

## Removidos

- `dados_externos/human_chats_sample.jsonl` — 52.467.916 bytes; derivação curada preservada em `dados_modulos_novos/voz_sft_curado.jsonl`; SHA-256 `ECD4A881316723DB498314BA1A2CA7DE849E6F0BDAB695D5BAD89F1B296A63F2`.
- `dados_modulos_novos/sinais_professor_natural_piloto.jsonl` — 3.792 bytes; piloto descartado por conter sinais inadequados/truncados; SHA-256 `7309DCD738D656B3A405FCB4DD2513859AAA9B314D93D9EA8FB63F3EBB48934F`.
- `dados_modulos_novos/sinais_professor_qwen_piloto.jsonl` — 3.557 bytes; piloto sem condicionamento natural, não usado no treino; SHA-256 `5A404128D0EF1C5BB2D7791EB2FE8C50AB1B6CA5DF9528AE7F19E4D014C718AE`.

## Preservados nesta etapa

- `dados_professor/deepseek_v4_flash_teacher_distillation.parquet` — fonte local do novo professor; 40.513 registros.
- `dados_modulos_novos/pensamento_think.jsonl` — reservado para o módulo pensamento.
- Benchmarks de farmacologia — somente validação.
- Tokenizers, dados ainda não incorporados, RAG, índices e checkpoints.

Regra: um dataset só pode ser removido após existir um artefato derivado ou checkpoint validado e uma entrada neste relatório.

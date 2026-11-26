# Project: Sandbox Generativo Voz BrotherTrue

## Architecture
- **Environment**: Isolated Sandbox at `/mnt/c/Users/Samsung/Downloads/brothertrue/sandbox_eval/`.
- **Core Repository Isolation**: Strict zero-modification policy for core modules:
  - `brothertrue.py`, `ode.py`, `neuroquimica.py`, `memoria.py`, `pensamento.py`, `palavras.py`, `ciclo_rem.py`, `voz.py`.
- **Runtime Governance**: `SandboxResourceGovernor` with `/proc/meminfo` fallback for Linux/WSL2 and `throttle()` per batch, restricted to 2-4 threads via `torch.set_num_threads(2)`.
- **Checkpoint Persistence**: Atomic checkpoints via `checkpoint_manager_safe.py` (`mkstemp` + `os.replace`, saving model, optimizer, scheduler, and Python/NumPy/PyTorch RNG states).
- **Teacher Model Integration**: Gemini PhD-level curation rejecting Fortran/truncated noise and synthesizing high-fidelity biocognitive conversational pairs (`unique_ratio_words >= 0.55`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Sandbox Directory Layout | Create isolated structure: dados, scripts, checkpoints, relatorios | M1 | R1 |
| 2 | Sandbox Resource Governor | Implement governor with Linux /proc/meminfo fallback and batch throttle | M1 | R4 |
| 3 | Baseline Diagnostic Audit | Objective pre-training audit of checkpoints_voz_lora_sft/best.pt | M1 | R1, R2 |
| 4 | Baseline Report Generation | Generate diagnostico_baseline.json with repetition and eos metrics | M1 | R2 |
| 5 | Review Queue Noise Filtering | Audit and discard corrupted review_queue items (Fortran, empty fields) | M2 | R3 |
| 6 | Teacher Data Synthesis | Synthesize rich biocognitive pairs in PT/EN with proper chat tokens | M2 | R3 |
| 7 | Dataset Manifest & SHA-256 | Generate manifest.json with line counts, token stats, and SHA-256 | M2 | R3 |
| 8 | SFT Training Harness | Train LoRA under ResourceGovernor and checkpoint_manager_safe | M3 | R4 |
| 9 | Atomic Safe Checkpoints | Save weights, optimizer, and RNG states atomically to best.pt/latest.pt | M3 | R4 |
| 10 | Post-Training Comparative Eval | Evaluate trained checkpoint against baseline with identical test suite | M4 | R5 |
| 11 | Mood Perturbation Test | Verify cross-attention stability under 11D mood perturbations | M4 | R1, R5 |
| 12 | Aval Técnico & Sign-off | Issue formal Google QA Blameless Postmortem and Technical Aval | M4 | R5, AGENTS.md |
| 13 | E2E Test Suite | Automated opaque-box validation of isolation, manifest, checkpoints, metrics | E2E Track | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Sandbox Infra & Baseline Audit | Setup sandbox_eval/, SandboxResourceGovernor, run baseline audit and generate diagnostico_baseline.json | none | IN_PROGRESS |
| M2 | Dataset Curation & Teacher Synthesis | Filter review_queue, synthesize train/val datasets, generate manifest.json with SHA-256 | M1 | PLANNED |
| M3 | Governed SFT Training | Implement train_sft_governed.py, train LoRA model on curated data, save safe checkpoints | M2 | PLANNED |
| M4 | Comparative Validation & Aval Técnico | Evaluate post-training model, run mood perturbation tests, generate comparative report and Parecer de Aval Técnico | M3 | PLANNED |
| E2E | E2E Testing Suite | Requirements-driven opaque-box verification test suite and publish TEST_READY.md | M1 | IN_PROGRESS |

## Interface Contracts
### SandboxResourceGovernor ↔ Training & Audit
- Method: `throttle() -> dict[str, float | bool]`
- Method: `ram_percent() -> float` (with `/proc/meminfo` POSIX fallback)
- Method: `process_cpu_percent() -> float`
- Invocation: Must be called at every training batch and audit iteration.

### Dataset ↔ Model & Tokenizer
- File: `sandbox_eval/dados/treino_voz_curado.jsonl`
- Format: JSONL with keys `prompt` and `response`
- Prompt format: `<|im_start|>user\n{pergunta}<|im_end|>\n<|im_start|>assistant\n`
- Sequence budget: `prompt_len + response_len <= 128`
- Tokenizer: `tokenizer_voz_novo_16k.json` (vocab size 16.384, special tokens: `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`)

### Checkpoints ↔ Evaluation Harness
- Checkpoint payload keys: `model`, `optimizer`, `scheduler`, `epoch`, `step`, `metrics`, `python_rng`, `numpy_rng`, `torch_rng`.
- Target files: `sandbox_eval/checkpoints/best.pt`, `sandbox_eval/checkpoints/latest.pt`.
- Loading: `torch.load(..., weights_only=False)`.

## Code Layout
```
sandbox_eval/
├── dados/
│   ├── manifest.json                  # Line counts, token stats, SHA-256 hashes
│   ├── treino_voz_curado.jsonl        # High-diversity curated SFT training pairs
│   └── val_curado.jsonl               # Independent validation pairs
├── scripts/
│   ├── sandbox_resource_governor.py   # Cooperatively throttled governor with Linux fallback
│   ├── audit_baseline.py              # Objective baseline audit of checkpoints_voz_lora_sft/best.pt
│   ├── train_sft_governed.py          # Governed training harness with checkpoint_manager_safe
│   ├── eval_comparative.py            # Comparative evaluation (pre vs post)
│   └── test_mood_perturbation.py      # Stability test under 11D mood vector perturbations
├── checkpoints/
│   ├── best.pt                        # Best validated checkpoint with full RNG states
│   └── latest.pt                      # Continuity checkpoint
├── relatorios/
│   ├── diagnostico_baseline.json      # Quantitative baseline audit metrics and samples
│   ├── avaliacao_pos_treino.json      # Post-training comparative metrics
│   └── parecer_aval_tecnico.md        # Google QA Lead Technical Aval & RCA
└── tests/
    └── test_e2e_sandbox.py            # Automated E2E verification test suite
```

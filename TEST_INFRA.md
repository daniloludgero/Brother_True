# Test Infrastructure & Specification: BrotherTrue Sandbox E2E Suite

**Document Version**: 1.0.0  
**Target Environment**: Linux WSL2 / POSIX (`/mnt/c/Users/Samsung/Downloads/brothertrue/sandbox_eval/`)  
**Standard**: Google SRE / Staff QA Engineering Quality Gates  
**Primary Suite Runner**: `sandbox_eval/tests/test_e2e_sandbox.py`  

---

## 1. Executive Summary & Architecture Overview

The BrotherTrue Biocognitive Sandbox (`sandbox_eval/`) establishes an isolated, hermetic evaluation and fine-tuning environment for the BrotherTrue Voz Transformer Decoder without modifying any core repository files.

To guarantee absolute compliance with the architectural invariants defined in `PROJECT.md` and `ORIGINAL_REQUEST.md`, this test infrastructure implements a **4-Tier Opaque-Box Test Architecture**:

```
+-----------------------------------------------------------------------------------+
|                           BrotherTrue E2E Test Suite                              |
|                       (sandbox_eval/tests/test_e2e_sandbox.py)                    |
+-----------------------------------------------------------------------------------+
         │
         ├──► [Tier 1: Strict Code Isolation]
         │      ├── Git Porcelain & Cryptographic SHA-256 Hash Matching
         │      └── Verification of 8 Untouched Core Engine Modules
         │
         ├──► [Tier 2: Data Governance & Manifest Integrity]
         │      ├── JSONL Format & Schema Validation (`prompt`, `response`)
         │      ├── Sequence Budget Checks (`prompt_len + resp_len <= 128`)
         │      └── Cryptographic SHA-256 & Exact Line Count Manifest Verification
         │
         ├──► [Tier 3: Runtime Governance & Checkpoint Persistence]
         │      ├── POSIX `/proc/meminfo` Fallback & CPU Throttling Validation
         │      ├── Restricted Threading (`torch.set_num_threads(2..4)`)
         │      └── Checkpoint Atomic Structure with RNG States (`python`, `numpy`, `torch`)
         │
         └──► [Tier 4: Generative Quality Acceptance]
                ├── Lexical Diversity Threshold (`unique_ratio_words >= 0.55`)
                ├── Formal Sentence Termination Rate (`ended_eos >= 85%`)
                └── Rejection of Degenerate Repetition Loops (e.g. `0.0208` token cycle)
```

---

## 2. Progressive Testability Model

The E2E test harness is designed to execute cleanly across the entire project lifecycle, respecting **Progressive Testability**:

| Milestone | Artifacts Produced | Test Tier Status | Progressive Behavior |
|:---|:---|:---|:---|
| **M1: Baseline Audit** | `sandbox_eval/`, `SandboxResourceGovernor`, `diagnostico_baseline.json` | Tier 1 (Active), Tier 3 Governor (Active) | Validates isolation, resource governor `/proc/meminfo` fallback, and baseline diagnostic structures. |
| **M2: Data Curation** | `sandbox_eval/dados/treino_voz_curado.jsonl`, `manifest.json` | Tier 2 (Fully Active) | Enforces strict JSONL schemas, line count verification, and SHA-256 cryptographic match. |
| **M3: SFT Training** | `sandbox_eval/checkpoints/best.pt`, `latest.pt` | Tier 3 Checkpoint (Fully Active) | Enforces atomic checkpoint layout, weights loading with `weights_only=False`, and complete RNG states. |
| **M4: Validation** | `sandbox_eval/relatorios/avaliacao_pos_treino.json`, `parecer_aval_tecnico.md` | Tier 4 (Fully Active) | Audits final metrics (`unique_ratio_words >= 0.55`, `ended_eos >= 85%`, absence of repetition collapse). |

If an artifact corresponding to a later milestone has not yet been generated, the test framework records an explicit informational notice or graceful skip, while running all available assertions and mathematical invariant validators. Once the milestone artifact is delivered, the test transition occurs automatically without modifying test code.

---

## 3. Tier-by-Tier Specification & Expected Output Derivation

### Tier 1: Strict Code Isolation
* **Authoritative Source**: `ORIGINAL_REQUEST.md` (R1, Acceptance Criteria: Isolamento Estrito do Código-Fonte); `PROJECT.md` § Architecture.
* **Invariant**: The 8 core repository files must never be modified under any circumstance:
  1. `brothertrue.py` (Baseline SHA-256: `e39e6d7317116d895c353e15ce1b1091c6747bc517c6ca9afc2d42d3ec5c1e7c`)
  2. `ode.py` (Baseline SHA-256: `0e664df3467a433e8fe9188b5a6a3f8d1f477c23aef03e201402081ab84a0355`)
  3. `neuroquimica.py` (Baseline SHA-256: `8bfb622cfd7fe8bba0c16cae88bd759d39a8b005b3189b2caaab92aa3557af4c`)
  4. `memoria.py` (Baseline SHA-256: `7f3e494ec22eb9880d3582419cbf77de95bc19903f497fd43c0f52999921250a`)
  5. `pensamento.py` (Baseline SHA-256: `09fbc04afe8050fb4e2a838eb9cff5370223f23efd283c0b16a772dc485240e8`)
  6. `palavras.py` (Baseline SHA-256: `6e722a42b50852925343a2b543734cf15e974a42c5813b86460ec3c8f085aa1e`)
  7. `ciclo_rem.py` (Baseline SHA-256: `5b4f2bdae2964e3e90f3c8f6823dfc37a3541479a8f0a0bd083fd61ce301c073`)
  8. `voz.py` (Baseline SHA-256: `503ebd3494af0e0a5fca40e249e705998eedeb2704df13772108b8379041c80b`)
* **Verification Logic**:
  - Dual-layer audit:
    - Layer A (Git Status): If `.git` metadata is present, invokes `git status --porcelain` to verify none of the 8 core files appear as modified (`M`) or deleted (`D`).
    - Layer B (Cryptographic Digest): In both Git and non-Git environments, reads the binary stream of each core file, computes `hashlib.sha256(content).hexdigest()`, and asserts exact identity against the verified baseline digests.

---

### Tier 2: Data Governance & Manifest Verification
* **Authoritative Source**: `ORIGINAL_REQUEST.md` (R3); `PROJECT.md` § Interface Contracts: Dataset <-> Model & Tokenizer; `AGENTS.md` (Data Retention & Review Queue Policies).
* **Target Files**:
  - `sandbox_eval/dados/treino_voz_curado.jsonl`
  - `sandbox_eval/dados/manifest.json`
  - (Optional) `sandbox_eval/dados/val_curado.jsonl`
* **Verification Logic**:
  1. **JSONL Format & Structure**: Every line in the dataset file must be valid JSON parseable into a dictionary with non-empty string fields `"prompt"` and `"response"`.
  2. **Special Tokens & Delimiters**: The `"prompt"` field must adhere strictly to ChatML / im format:
     `^<\|im_start\|>user\n[\s\S]+?<\|im_end\|>\n<\|im_start\|>assistant\n$`
  3. **Noise Filtering**: Neither prompt nor response may contain uncurated Fortran/C keywords (e.g., `SUBROUTINE CSCFG`, `claw-compiler`), nor empty response bodies.
  4. **Line Count & SHA-256 Verification**:
     - Manifest must record the file name, line count, and SHA-256 hash.
     - The test reads the dataset file, counts `\n` line delimiters, computes the SHA-256 hex digest, and asserts identical match (case-insensitive) with `manifest.json`.

---

### Tier 3: Runtime Governance & Checkpoint Integrity
* **Authoritative Source**: `ORIGINAL_REQUEST.md` (R4); `PROJECT.md` § Interface Contracts: SandboxResourceGovernor & Checkpoints; `AGENTS.md` § 2 & 3.
* **Component 3A: SandboxResourceGovernor**:
  - Script: `sandbox_eval/scripts/sandbox_resource_governor.py`.
  - Class: `SandboxResourceGovernor(policy=ResourcePolicy(cpu_limit_percent=75.0, ram_limit_percent=60.0), num_threads=2)`.
  - Method `ram_percent()`: Must execute without raising exceptions, falling back to `/proc/meminfo` in Linux/WSL2 environments, returning a float strictly between $0.0\%$ and $100.0\%$.
  - Method `process_cpu_percent()`: Must return a non-negative float $\le 100.0\%$.
  - Method `throttle()`: Must return a dictionary containing keys `{"cpu_percent", "ram_percent", "paused"}`.
* **Component 3B: Checkpoint Layout & RNG States**:
  - Module: `checkpoint_manager_safe.py` and checkpoint files `sandbox_eval/checkpoints/best.pt` / `latest.pt`.
  - Verification:
    - Atomicity: Test verifies that `save_checkpoint` writes to a temporary file (`.tmp`) via `tempfile.mkstemp` and performs atomic replacement via `os.replace`.
    - Payload Schema: Checkpoint payload must contain keys:
      `['model', 'optimizer', 'scheduler', 'epoch', 'step', 'metrics', 'python_rng', 'numpy_rng', 'torch_rng']`.
    - State Preservation: `python_rng` must be a valid tuple from `random.getstate()`, `numpy_rng` must be a valid tuple/dict from `np.random.get_state()`, and `torch_rng` must be a tensor of type `torch.ByteTensor`.

---

### Tier 4: Generative Quality Acceptance
* **Authoritative Source**: `ORIGINAL_REQUEST.md` (R1, R2, R5, Acceptance Criteria: Validação de Resultados e Redução de Inconstâncias); `relatorios/diagnostico_colapso_saida_voz.json`; `relatorios/voz_eos_comparacao.json`.
* **Historical Failure Baselines (To Reject)**:
  - Repetition Loop Pathology: Repetitive quotation marks or single words (`" "" "" "` or `"i i i i"`) yielding `unique_ratio_words = 0.020833` ($1 / 48$).
  - Premature Truncation Pathology: Immediate `<eos>` emission at token 1 (`completion: ""` with `new_tokens: 1`), collapsing verbal response.
  - Runaway Pathology: Reaching token horizon (`max_new=48`) without `<eos>` termination (`ended_eos: false`).
* **Homologated Quality Gates**:
  1. **Lexical Diversity**: `unique_ratio_words >= 0.55`, where:
     $$\text{unique\_ratio\_words} = \frac{|\text{set}(\text{words})|}{\max(1, |\text{words}|)}$$
  2. **EOS Termination Rate**: `ended_eos >= 85%` ($0.85$), where `ended_eos = bool(tokens[-1] == eos_token_id)`.
  3. **Absence of Repetition Collapse**: Assert zero generations exhibit `unique_ratio_words <= 0.10` or the historical $0.0208$ cycle.
  4. **Post-Training Report Audit**: When `sandbox_eval/relatorios/avaliacao_pos_treino.json` is generated, verify that the aggregated metrics satisfy these quality gates.

---

## 4. Test Runner Invocation

The test suite is located at `sandbox_eval/tests/test_e2e_sandbox.py`. It is fully self-contained and supports standard execution across Linux WSL2 and POSIX environments.

### Primary Standalone Runner (Zero External Test Dependencies)
```bash
python3 sandbox_eval/tests/test_e2e_sandbox.py
```
*Outputs detailed per-tier test execution, progress summary, and exit code `0` on success.*

### Pytest Runner (If Pytest is Installed in Virtual Environment)
```bash
pytest -v sandbox_eval/tests/test_e2e_sandbox.py
```

### Unittest Module Discovery
```bash
python3 -m unittest discover -s sandbox_eval/tests -p "test_*.py" -v
```

---

## 5. Failure Triage & Remediation Matrix

| Symptom / Test Failure | Root Cause Analysis | Remediation Action |
|:---|:---|:---|
| **Tier 1 Fails**: Core file hash mismatch | A script in `sandbox_eval/` or root altered a protected file (`brothertrue.py`, `voz.py`, etc.). | Revert modifications to core files immediately; maintain all logic within `sandbox_eval/`. |
| **Tier 2 Fails**: SHA-256 mismatch in `manifest.json` | The JSONL file was edited after manifest creation, or line endings (`\r\n` vs `\n`) drifted. | Recompute SHA-256 using binary stream (`rb`) and update `sandbox_eval/dados/manifest.json`. |
| **Tier 2 Fails**: Invalid ChatML syntax | Prompt is missing `<\|im_start\|>user` or `<\|im_start\|>assistant\n`. | Align generation template with `tokenizer_voz_novo_16k.json` chat conventions. |
| **Tier 3 Fails**: `ram_percent` returns `0.0%` | Linux environment failed to parse `/proc/meminfo`. | Check permissions and existence of `/proc/meminfo` on WSL2 host. |
| **Tier 3 Fails**: Checkpoint missing `torch_rng` | Checkpoint saved with legacy `checkpoint_manager.py` instead of `checkpoint_manager_safe.py`. | Adopt `checkpoint_manager_safe.save_checkpoint()` saving full RNG tuples. |
| **Tier 4 Fails**: `unique_ratio_words < 0.55` | Generative sampling fell into repetition loop or training dataset contains redundant responses. | Increase sampler repetition penalty ($1.20 - 1.35$) or curate higher diversity in teacher dataset. |
| **Tier 4 Fails**: `ended_eos < 85%` | Model runs away past context or emits premature `<eos>` before completion. | Enforce minimum new tokens (`min_new_tokens=6`) and verify `<eos>` placement in SFT targets. |

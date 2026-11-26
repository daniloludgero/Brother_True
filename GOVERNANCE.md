# Governance & Epistemic Stewardship (`GOVERNANCE.md`)

## 1. Leadership Model (BDFL $\to$ Technical Steering Committee)

**BrotherTrue** operates under a **Benevolent Dictator For Life (BDFL)** governance structure led by its creator (**Danilo Emanuel Ludgero Costa**), designed to transition smoothly to a 3–5 member **Technical Steering Committee (TSC)** as external academic and systems contributors mature.

## 2. Non-Negotiable Architectural & Epistemic Laws

Inspired by both [`AGENTS.md`](AGENTS.md) and the **Secretaria de Ensino** governance protocol:
1. **8-Node Pipeline & Inheritance Preservation**: Every PR touching `brothertrue.py`, `voz.py`, or `models/` MUST pass `python3 tests/conferir_arquitetura.py` (`CONFORME`).
2. **Authorial Decoder Identity**: External LLMs act as teachers, reviewers, and benchmarks; the student (`Voz` / `VozDistribuida` / `OwnVoice`) remains an authorial neural module.
3. **Prior Anchor & Honest Ablations**: Distilled updates or training runs must report both **gains and losses** on held-out evaluation sets (`aula ∩ prova == ∅` via SHA-256) and default to `enabled_by_default: false` until explicit human sign-off.
4. **Refusal of Overclaiming (`CLAIM_STRONGER_THAN_SOURCE`)**: Documentation, PR descriptions, and UI strings must never conflate proxy metrics (`val_loss`, `<eos>`, `unique_ratio_words`) with verified human-level semantic fluency.

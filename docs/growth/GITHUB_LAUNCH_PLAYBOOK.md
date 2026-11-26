# GitHub Launch & Community Distribution Playbook (`GITHUB_LAUNCH_PLAYBOOK.md`)

## 1. Repository Header Configuration (`.github/REPO_METADATA.json`)

- **Description (`<= 120` chars)**:
  `Open-system biocognitive architecture coupling Qualia ODEs & Hill nM kinetics to an authorial decoder with RAG gating.`
- **Website**: `https://daniloludgero.github.io/Brother_True/`
- **Topics (8 focused tags)**:
  `biocognitive-architecture`, `neural-ode`, `cognitive-architecture`, `rag`, `chain-of-thought`, `neuro-symbolic-ai`, `pytorch`, `python`
- **Social Preview Image**: Export `docs/assets/social-preview.svg` (`1280×640`) to PNG when uploading to *Settings → General → Social Preview*.

---

## 2. "Show HN" Launch Protocol (Hacker News)

- **Recommended Timing**: Tuesday or Wednesday, 08:00 EST (10:00 BRT).
- **Title**:
  `Show HN: BrotherTrue – Coupling continuous Qualia ODEs and Hill receptor kinetics to a from-scratch decoder`
- **Author First Comment (Post within 60 seconds)**:
  1. **Why we built it**: Stateless LLMs reset emotionally and epistemically on every turn, and often prune uncertainty into confident hallucinations. We wanted an open-book laboratory testing continuous affective ODEs + biophysical Hill occupancy ($nM$) coupled to an explicit RAG Evidence Gate.
  2. **How the 8-node pipeline works**: Message $\to$ Semantic Encoder $\to$ `QualiaODE` (with mandatory `apply_collapse` damping on $E > E_{\text{crit}}$) + 8 Hill channels + SQLite LTP $\to$ causal $t-1$ `Pensamento` (`loss_p`) + `Palavras` (`KL` surprise) $\to$ pre-decoder `BiocognitiveReasoning` $\to$ `EvidenceGate` $\to$ deterministic `<thought>` projection $\to$ authorial `Voz`/`OwnVoice` decoder $\to$ tag-stripped verbal output.
  3. **Radical honesty on what works today vs. what doesn't**: The 8-node pipeline AST, 20D state contracts, LayerNorm/SDPA causal mask fixes (`31/31` tests), and interactive browser simulator work today. Natural human-level Portuguese fluency from our from-scratch student decoder is **not** yet established—we explicitly rejected using `unique_ratio_words = 0.92` or mock teacher wrappers to fake fluency while cloud pre-training is underway.
  4. **Try in 10 seconds**: Link to the Interactive Web Lab (`docs/index.html`) + `python3 tests/conferir_arquitetura.py`.

---

## 3. Reddit & Technical Communities

- **Subreddits**: `r/MachineLearning` (`[P]` tag), `r/LocalLLaMA`, `r/Python`, `r/CognitiveScience`.
- **Format**: Deep-dive text post focusing 80% on the **4 engineering bugs uncovered during internal audits** (premature `<eos>`, proxy lexical diversity vs. real semantics, 1D `LayerNorm` zero-init in `VozDistribuida`, and SDPA `is_causal` mask interaction) and 20% on repository links at the footer.

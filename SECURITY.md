# Security & Epistemic Safety Policy (`SECURITY.md`)

## 1. Research Scope & Deployment Boundaries

**BrotherTrue (`v0.5.0-alpha`)** is an experimental biocognitive architecture and open-core verification workbench. It is **NOT** certified for clinical, medical, pharmacological, legal, or safety-critical decision-making.
- Biophysical Hill equations in `farmacologia.py` model computational receptor occupancy dynamics inside a synthetic cognitive organism; they **must never** be interpreted as clinical dosing or medical guidance.
- Channel separation (`DeepSeekCoTParser` and `ResponseEnvelope`) strips `<thought>` tags and control tokens deterministically, but does not guarantee semantic infallibility of unreviewed model outputs.

## 2. Threat Surface Monitored

We treat the following vectors as security-relevant defects:
1. **Prompt / RAG Tag-Channel Injection**: Crafted passages in the Vector DB or LTP attempting to inject `</thought>`, `<|im_start|>`, or role-spoofing tokens past `EvidenceGate._sanitizar()` or `contracts/rendering.py`.
2. **Evidence Provenance Bypass**: Promotion of unreviewed or below-threshold retrieval items to `"supported"` verdicts in `grounded` mode (`CLAIM_STRONGER_THAN_SOURCE`).
3. **Unsafe Deserialization & SQLite Corruption**: Unverified `torch.load` payloads from untrusted sources or concurrent unclosed writes to `qualia_memory.db` / `qualia_ltp.db`.
4. **Resource Exhaustion**: Circumvention of `ResourceGovernor` / `HardwareSentry` host-disk safety floors.

## 3. Reporting a Vulnerability

Please report security or channel-boundary bypasses privately via **GitHub Private Vulnerability Reporting** on the repository's **Security** tab (do not open public issues with exploit payloads prior to triage). Include a minimal deterministic reproduction script (`HF_HUB_OFFLINE=1`). We target initial triage acknowledgement within **72 hours**.

# Contributing to BrotherTrue (`CONTRIBUTING.md`)

Thank you for contributing to **BrotherTrue**. To keep the biocognitive engine reproducible, honest, and lightweight on constrained hardware, please follow these rules:

## 1. Mandatory Pre-PR Verification Gate

Run these three fast offline checks before opening any Pull Request:

```bash
# 1. Static AST verification of the 8-node pipeline and nn.Module inheritance
python3 tests/conferir_arquitetura.py

# 2. Standard-library v1 contracts (20D state, evidence modes, channel boundaries)
python3 -m unittest discover -s experimentos/industrial_readiness_2026_10_25/tests -p "test_*contract*.py" -v

# 3. Syntax & DTCG design-token check
python3 -m py_compile brothertrue.py voz.py models/evidence_gate.py models/thought_renderer.py
```

## 2. Engineering & Hardware Constraints

- **Offline-First & Thread Cap**: Test scripts must set `os.environ["HF_HUB_OFFLINE"] = "1"` and `torch.set_num_threads(2)`.
- **Zero Binary Bloat**: Never commit `.pt` checkpoints, `.db` SQLite databases, raw datasets, or third-party binary sprites/assets to Git. Weights and corpora belong in Google Drive / HuggingFace Hub accompanied by SHA-256 manifests.
- **Design System Discipline**: Any update to `docs/` or SVGs must consume tokens from [`docs/design/tokens.dtcg.json`](docs/design/tokens.dtcg.json) (`>= 4.5:1` WCAG AA contrast, `tabular-nums` on telemetry).
- **Developer Certificate of Origin (DCO)**: Sign off commits with `git commit -s` (`Signed-off-by: Name <email>`).

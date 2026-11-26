"""Build a small, explicit source delivery; never copy weights or live SQLite.

Generated ZIP/notebook/manifest are delivery artifacts, not model checkpoints.
This program uses only the standard library and may run on the host.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
DELIVERY_NAME = "BrotherTrue_experimental_2026-10-26"
CORE = [
    "AGENTS.md", "voz_distribuida.py", "models/__init__.py",
    "models/biocognitive_projector.py", "models/qwen_adapter.py", "models/cot_parser.py",
    "ode.py", "neuroquimica.py", "farmacologia.py", "pensamento.py", "palavras.py",
    "memoria.py", "long_term_memory.py", "ciclo_rem.py", "semantica.py",
    "resource_governor.py", "checkpoint_manager_safe.py",
    "tokenizer_voz_novo_16k.json",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    out = EXPERIMENT / "deliverables"
    out.mkdir(exist_ok=True)
    paths = [REPO / name for name in CORE]
    for section in ("contracts", "modeling", "tests"):
        paths.extend(sorted((EXPERIMENT / section).glob("*.py")))
        paths.extend(sorted((EXPERIMENT / section).glob("*.md")))
    paths.extend([EXPERIMENT / "training/own_decoder_runner.py", EXPERIMENT / "README.md",
                  EXPERIMENT / "ARTIFACT_CONTRACT.md", EXPERIMENT / "evaluation/findings.md",
                  EXPERIMENT / "evaluation/PROTOCOL.md"])
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("All implementation files must exist before packaging")
    content = {path.relative_to(REPO).as_posix(): path.read_bytes() for path in paths}
    if sum(map(len, content.values())) > 4 * 1024**2:
        raise ValueError("Source delivery exceeds 4 MiB; inspect its explicit inputs")
    manifest = {
        "schema_version": 1, "purpose": "own_decoder_contract_verification",
        "student_initialization": "random_from_scratch",
        "teacher_weights_included": False, "model_weights_included": False,
        "training_executed": False, "commercial_readiness": "not_established",
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in content.items()},
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    archive = out / "brothertrue_source_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in sorted(content.items()):
            target.writestr(name, data)
        target.writestr("SOURCE_MANIFEST.json", manifest_bytes)
    bundle_sha = digest(archive.read_bytes())
    (out / "source_manifest.json").write_bytes(manifest_bytes)
    bootstrap = f'''from google.colab import drive
from pathlib import Path
import hashlib, json, os, shutil, sys, tempfile, zipfile
drive.mount('/content/drive')
if sys.platform != 'linux' or not Path('/content').is_dir():
    raise RuntimeError('Google Colab required')
base = Path('/content/drive/MyDrive/{DELIVERY_NAME}')
bundle = base / 'brothertrue_source_bundle.zip'
expected_sha256 = '{bundle_sha}'
if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected_sha256:
    raise RuntimeError('Source ZIP hash mismatch')
if shutil.disk_usage('/content').free < 4.5 * 1024**3:
    raise RuntimeError('Colab disk below 4.5 GiB')
work = Path(tempfile.mkdtemp(prefix='brothertrue_', dir='/content'))
with zipfile.ZipFile(bundle) as archive:
    for item in archive.infolist():
        target = (work / item.filename).resolve()
        if work not in target.parents or item.file_size > 4 * 1024**2:
            raise RuntimeError('Invalid bundle entry')
    archive.extractall(work)
manifest = json.loads((work / 'SOURCE_MANIFEST.json').read_text())
for name, meta in manifest['files'].items():
    if hashlib.sha256((work/name).read_bytes()).hexdigest() != meta['sha256']:
        raise RuntimeError('Source hash mismatch: ' + name)
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(work))
experiment = work / 'experimentos/industrial_readiness_2026_10_25'
sys.path.insert(0, str(experiment))
print('Verified sources:', len(manifest['files']), 'workdir:', work)
'''
    verify = '''import subprocess, time
import torch
torch.set_num_threads(2)
results = []
for filename in ('test_contracts.py', 'test_runner_contract.py', 'test_own_voice.py'):
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-B', str(experiment/'tests'/filename)],
                            cwd=work, capture_output=True, text=True)
    print(result.stdout, result.stderr)
    results.append({'suite':filename, 'exit_code':result.returncode,
                    'elapsed_seconds':time.monotonic()-started,
                    'stdout':result.stdout, 'stderr':result.stderr})
report = {'tests':results, 'torch_version':torch.__version__,
          'cuda_available':torch.cuda.is_available(),
          'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'source_zip_sha256':expected_sha256, 'training_executed':False,
          'verification_passed':all(r['exit_code'] == 0 for r in results),
          'commercial_readiness':'not_established'}
fd, temp_report = tempfile.mkstemp(prefix='verification_', suffix='.json', dir=base)
with os.fdopen(fd, 'w') as handle:
    json.dump(report, handle, indent=2)
report_path = base / (Path(temp_report).stem + '_complete.json')
os.replace(temp_report, report_path)
print('Saved verification:', report_path)
if not report['verification_passed']:
    raise RuntimeError('Verification failed; evidence preserved; no training or promotion')
'''
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            cell("markdown", "# BrotherTrue: verificação do decoder próprio\n\nFontes isoladas, pesos inicializados do zero. Este notebook executa regressões pequenas no Colab e salva o relatório no Drive. Não treina um modelo comercial. O ZIP deve estar na mesma pasta do Drive que este notebook.\n"),
            cell("code", bootstrap),
            cell("code", verify),
            cell("markdown", "## Próxima etapa\n\nApós testes aprovados: produzir traços reais do núcleo; revisar corpus com professores; congelar treino/validação/teste; medir custo do pré-treinamento; executar piloto do `training/own_decoder_runner.py` conforme sua CLI. O checkpoint do aluno inclui todos os pesos. Os arquivos históricos de SFT Qwen não são o caminho de treinamento deste aluno.\n"),
        ],
    }
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"brothertrue-{index}"
    (out / "brothertrue_own_decoder_verification.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "delivery_hashes.json").write_text(json.dumps({
        path.name: digest(path.read_bytes()) for path in (archive, out/"source_manifest.json",
                                                       out/"brothertrue_own_decoder_verification.ipynb")
    }, indent=2), encoding="utf-8")
    print(json.dumps({"bundle_bytes": archive.stat().st_size, "sha256": bundle_sha,
                      "directory": str(out), "files": len(content)}))


if __name__ == "__main__":
    build()

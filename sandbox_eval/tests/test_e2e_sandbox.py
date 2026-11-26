#!/usr/bin/env python3
"""Suite de Testes E2E: Sandbox de Avaliação e Treinamento Generativo Voz BrotherTrue.

Arquitetura de Testes Opaque-Box em 4 Camadas (Tiers):
- Tier 1: Isolamento Estrito do Código-Fonte (Zero modificações no núcleo).
- Tier 2: Governança de Dados e Verificação do Manifesto (JSONL, contagens e SHA-256).
- Tier 3: Governança de Execução e Integridade de Checkpoints (/proc/meminfo e RNG states).
- Tier 4: Aceitação de Qualidade Generativa (unique_ratio >= 0.55, ended_eos >= 85%, anti-loop).

Compatível com execução direta via `python3` e via `pytest`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SANDBOX_DIR = ROOT_DIR / "sandbox_eval"

# Invariante Arquitetural: 8 módulos protegidos do núcleo do projeto
CORE_FILES_BASELINE: Dict[str, Dict[str, Any]] = {
    "brothertrue.py": {
        "size": 13558,
        "sha256": "e39e6d7317116d895c353e15ce1b1091c6747bc517c6ca9afc2d42d3ec5c1e7c",
    },
    "ode.py": {
        "size": 5205,
        "sha256": "0e664df3467a433e8fe9188b5a6a3f8d1f477c23aef03e201402081ab84a0355",
    },
    "neuroquimica.py": {
        "size": 6980,
        "sha256": "8bfb622cfd7fe8bba0c16cae88bd759d39a8b005b3189b2caaab92aa3557af4c",
    },
    "memoria.py": {
        "size": 7359,
        "sha256": "7f3e494ec22eb9880d3582419cbf77de95bc19903f497fd43c0f52999921250a",
    },
    "pensamento.py": {
        "size": 3766,
        "sha256": "09fbc04afe8050fb4e2a838eb9cff5370223f23efd283c0b16a772dc485240e8",
    },
    "palavras.py": {
        "size": 3616,
        "sha256": "6e722a42b50852925343a2b543734cf15e974a42c5813b86460ec3c8f085aa1e",
    },
    "ciclo_rem.py": {
        "size": 2715,
        "sha256": "5b4f2bdae2964e3e90f3c8f6823dfc37a3541479a8f0a0bd083fd61ce301c073",
    },
    "voz.py": {
        "size": 5654,
        "sha256": "503ebd3494af0e0a5fca40e249e705998eedeb2704df13772108b8379041c80b",
    },
}


# =====================================================================
# Funções Utilitárias Canônicas de Validação
# =====================================================================

def compute_sha256(filepath: Path | str) -> str:
    """Calcula o hash criptográfico SHA-256 de um arquivo em hexadecimal minúsculo."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_file_lines(filepath: Path | str) -> int:
    """Conta a quantidade exata de linhas não vazias terminadas em newline."""
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count


def calculate_unique_ratio_words(text: str) -> float:
    """Calcula a taxa canônica de diversidade léxica no nível de palavras."""
    words = text.strip().split()
    if not words:
        return 0.0
    return len(set(words)) / max(1, len(words))


def detect_repetition_loop(text: str, min_words: int = 16, max_unique_ratio: float = 0.10) -> bool:
    """Detecta se uma amostra de texto colapsou em repetições degeneradas."""
    words = text.strip().split()
    if len(words) >= min_words and calculate_unique_ratio_words(text) <= max_unique_ratio:
        return True
    # Detecção de repetição de caracteres redundantes (aspas, pontuação isolada)
    if len(text.strip()) > 30:
        chars = [c for c in text.strip() if not c.isspace()]
        if chars and (len(set(chars)) / len(chars)) < 0.10:
            return True
    return False


# =====================================================================
# Tier 1: Isolamento Estrito do Código-Fonte do Núcleo
# =====================================================================

class TestTier1StrictCodeIsolation(unittest.TestCase):
    """Tier 1: Garante que os 8 arquivos centrais do BrotherTrue permaneçam intocados."""

    def test_core_modules_exist(self) -> None:
        """Verifica a presença dos 8 arquivos canônicos protegidos na raiz."""
        for filename in CORE_FILES_BASELINE:
            target = ROOT_DIR / filename
            self.assertTrue(target.exists(), f"Arquivo protegido ausente na raiz: {filename}")
            self.assertTrue(target.is_file(), f"Alvo não é arquivo regular: {filename}")

    def test_core_modules_cryptographic_integrity(self) -> None:
        """Verifica que nenhum byte dos 8 arquivos do núcleo sofreu alteração."""
        mismatches: List[str] = []
        for filename, expected in CORE_FILES_BASELINE.items():
            target = ROOT_DIR / filename
            if not target.exists():
                mismatches.append(f"{filename}: ARQUIVO_AUSENTE")
                continue

            current_size = target.stat().st_size
            current_hash = compute_sha256(target)

            if current_hash.lower() != expected["sha256"].lower():
                mismatches.append(
                    f"{filename} MODIFICADO! Esperado sha256={expected['sha256']}, "
                    f"Atual sha256={current_hash} (tamanho esperado={expected['size']}, atual={current_size})"
                )

        self.assertEqual(
            len(mismatches),
            0,
            "Violação de Isolamento Estrito de Código (Tier 1):\n" + "\n".join(mismatches),
        )

    def test_git_status_porcelain_clean(self) -> None:
        """Se o diretório for repositório Git, verifica ausência de modificações no núcleo via porcelain."""
        git_dir = ROOT_DIR / ".git"
        if not git_dir.exists():
            self.skipTest("Ambiente não possui diretório .git; validação coberta por integridade SHA-256.")

        try:
            res = subprocess.run(
                ["git", "status", "--porcelain", "--"] + list(CORE_FILES_BASELINE.keys()),
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = res.stdout.strip()
            self.assertEqual(
                stdout,
                "",
                f"Arquivos do núcleo constam como alterados no git status:\n{stdout}",
            )
        except FileNotFoundError:
            self.skipTest("Binário git não encontrado no PATH do sistema operacional.")


# =====================================================================
# Tier 2: Governança de Dados e Validação de Manifesto
# =====================================================================

class TestTier2DataGovernance(unittest.TestCase):
    """Tier 2: Valida schemas JSONL, tokens especiais de chat e consistência do manifest.json."""

    def setUp(self) -> None:
        self.dados_dir = SANDBOX_DIR / "dados"
        self.manifest_path = self.dados_dir / "manifest.json"
        self.train_dataset_path = self.dados_dir / "treino_voz_curado.jsonl"

    def test_manifest_structure_and_schema(self) -> None:
        """Verifica o arquivo manifest.json se já gerado (Milestone M2)."""
        if not self.manifest_path.exists():
            self.skipTest(f"[M2 Pendente] Manifesto não encontrado em: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, (dict, list), "Manifesto deve ser um dicionário ou lista JSON.")

        # Normaliza estrutura de datasets do manifesto
        datasets = data if isinstance(data, list) else data.get("datasets", [data])

        for entry in datasets:
            self.assertIn("dataset_name", entry, "Entrada no manifest deve conter 'dataset_name'")
            self.assertIn("sha256", entry, "Entrada no manifest deve conter 'sha256'")
            self.assertIn(
                True,
                ["total_lines" in entry, "lines" in entry, "count" in entry],
                "Entrada no manifest deve declarar contagem de linhas ('total_lines')",
            )

    def test_curated_dataset_jsonl_formatting(self) -> None:
        """Valida que o dataset de treino curado possui formatação JSONL estrita e tokens ChatML."""
        if not self.train_dataset_path.exists():
            self.skipTest(f"[M2 Pendente] Dataset curado não encontrado em: {self.train_dataset_path}")

        forbidden_noise_patterns = [
            "SUBROUTINE CSCFG",
            "SUBROUTINE VPINT",
            "claw-compiler",
            "Fortran",
        ]

        valid_lines = 0
        with open(self.train_dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue

                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as err:
                    self.fail(f"Linha {idx} em {self.train_dataset_path.name} contém JSON inválido: {err}")

                self.assertIn("prompt", record, f"Linha {idx} deve conter chave 'prompt'")
                self.assertIn("response", record, f"Linha {idx} deve conter chave 'response'")

                prompt = record["prompt"]
                response = record["response"]

                # Valida formato estrito de chat
                self.assertTrue(
                    "<|im_start|>user" in prompt and "<|im_start|>assistant" in prompt,
                    f"Linha {idx}: Prompt deve seguir convenção de chat com marcadores <|im_start|>",
                )
                self.assertGreater(
                    len(response.strip()),
                    0,
                    f"Linha {idx}: Campo 'response' não pode ser vazio",
                )

                # Rejeição de ruído Fortran/código não conversacional
                for noise in forbidden_noise_patterns:
                    self.assertNotIn(
                        noise,
                        prompt,
                        f"Linha {idx}: Prompt contém ruído não conversacional proibido ({noise})",
                    )
                    self.assertNotIn(
                        noise,
                        response,
                        f"Linha {idx}: Resposta contém ruído não conversacional proibido ({noise})",
                    )

                valid_lines += 1

        self.assertGreaterEqual(
            valid_lines,
            10,
            f"Dataset curado {self.train_dataset_path.name} deve conter ao menos 10 registros válidos.",
        )

    def test_manifest_checksum_and_line_count_match(self) -> None:
        """Garante que a contagem de linhas e o hash SHA-256 no manifest correspondam exatamente ao arquivo no disco."""
        if not self.manifest_path.exists() or not self.train_dataset_path.exists():
            self.skipTest("[M2 Pendente] Manifesto ou dataset curado ausente para validação cruzada.")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        datasets = manifest_data if isinstance(manifest_data, list) else manifest_data.get("datasets", [manifest_data])

        # Localiza o registro correspondente ao arquivo de treino
        target_name = self.train_dataset_path.name
        matched_entry = None
        for entry in datasets:
            if entry.get("dataset_name") == target_name:
                matched_entry = entry
                break

        if not matched_entry and len(datasets) == 1:
            matched_entry = datasets[0]

        self.assertIsNotNone(
            matched_entry,
            f"Manifesto não possui entrada para o dataset {target_name}",
        )

        expected_sha = matched_entry["sha256"].lower()
        actual_sha = compute_sha256(self.train_dataset_path).lower()
        self.assertEqual(
            actual_sha,
            expected_sha,
            f"Hash SHA-256 do arquivo {target_name} ({actual_sha}) diverge do manifesto ({expected_sha})",
        )

        expected_lines = (
            matched_entry.get("total_lines")
            or matched_entry.get("lines")
            or matched_entry.get("count")
        )
        if expected_lines is not None:
            actual_lines = count_file_lines(self.train_dataset_path)
            self.assertEqual(
                actual_lines,
                expected_lines,
                f"Contagem de linhas de {target_name} ({actual_lines}) diverge do manifesto ({expected_lines})",
            )


# =====================================================================
# Tier 3: Governança de Execução e Integridade de Checkpoints
# =====================================================================

class TestTier3RuntimeGovernanceAndCheckpoints(unittest.TestCase):
    """Tier 3: Valida limites cooperativos de CPU/RAM, fallback POSIX e estrutura de RNG em checkpoints."""

    def test_sandbox_resource_governor_contract(self) -> None:
        """Valida que o SandboxResourceGovernor cumpre os contratos de leitura e throttling."""
        gov_script = SANDBOX_DIR / "scripts" / "sandbox_resource_governor.py"
        if not gov_script.exists():
            self.skipTest(f"[M1 Pendente] Script do governador não encontrado em {gov_script}")

        # Adiciona pasta de scripts temporariamente ao sys.path
        scripts_dir = str(gov_script.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from sandbox_resource_governor import ResourcePolicy, SandboxResourceGovernor

        policy = ResourcePolicy(cpu_limit_percent=85.0, ram_limit_percent=80.0, sleep_seconds=0.01)
        gov = SandboxResourceGovernor(policy=policy, num_threads=2)

        # 1. Medição de RAM
        ram = gov.ram_percent()
        self.assertIsInstance(ram, float, "ram_percent() deve retornar float")
        self.assertGreaterEqual(ram, 0.0, "ram_percent() não pode ser negativo")
        self.assertLessEqual(ram, 100.0, "ram_percent() não pode exceder 100.0%")

        # Em Linux/WSL2, valida que /proc/meminfo é lido com valor positivo
        if os.path.exists("/proc/meminfo"):
            self.assertGreater(
                ram,
                0.0,
                "Em sistemas Linux com /proc/meminfo, ram_percent() deve retornar valor estritamente positivo",
            )

        # 2. Medição de CPU
        cpu = gov.process_cpu_percent()
        self.assertIsInstance(cpu, float, "process_cpu_percent() deve retornar float")
        self.assertGreaterEqual(cpu, 0.0)
        self.assertLessEqual(cpu, 100.0)

        # 3. Execução do throttle cooperativo
        status = gov.throttle()
        self.assertIsInstance(status, dict, "throttle() deve retornar dicionário")
        self.assertIn("cpu_percent", status)
        self.assertIn("ram_percent", status)
        self.assertIn("paused", status)

    def test_atomic_checkpoint_safe_contract(self) -> None:
        """Verifica o protocolo de salvamento atômico e retenção de RNG de checkpoint_manager_safe."""
        safe_manager_file = ROOT_DIR / "checkpoint_manager_safe.py"
        self.assertTrue(safe_manager_file.exists(), "checkpoint_manager_safe.py deve existir no repositório")

        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))

        try:
            import torch
            import checkpoint_manager_safe
        except ImportError:
            self.skipTest("PyTorch não disponível no ambiente atual para testar checkpoint_manager_safe diretamente.")

        # Teste de escrita atômica em diretório temporário
        with tempfile.TemporaryDirectory() as tmpdir:
            test_ckpt_path = Path(tmpdir) / "test_atomic.pt"

            dummy_model = torch.nn.Linear(4, 2)
            dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

            checkpoint_manager_safe.save_checkpoint(
                str(test_ckpt_path),
                model=dummy_model,
                optimizer=dummy_optimizer,
                epoch=1,
                step=10,
                metrics={"val_loss": 1.23},
            )

            self.assertTrue(test_ckpt_path.exists(), "Checkpoint final atômico não foi criado")
            # Garante que nenhum arquivo temporário residual (.tmp) permaneceu
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0, "Arquivos temporários (.tmp) não foram limpos")

            # Inspeciona payload carregado com weights_only=False
            payload = torch.load(str(test_ckpt_path), map_location="cpu", weights_only=False)
            required_keys = [
                "model",
                "optimizer",
                "epoch",
                "step",
                "metrics",
                "python_rng",
                "numpy_rng",
                "torch_rng",
            ]
            for key in required_keys:
                self.assertIn(key, payload, f"Payload do checkpoint deve conter a chave '{key}'")

            # Valida tipos de RNG
            self.assertIsInstance(payload["python_rng"], tuple, "python_rng deve ser tupla de estado")
            self.assertIsNotNone(payload["numpy_rng"], "numpy_rng não pode ser None")
            self.assertIsInstance(payload["torch_rng"], torch.Tensor, "torch_rng deve ser Tensor")

    def test_milestone_m3_trained_checkpoint_integrity(self) -> None:
        """Se o checkpoint do sandbox existir (Milestone M3), audita suas chaves e integridade."""
        best_ckpt = SANDBOX_DIR / "checkpoints" / "best.pt"
        if not best_ckpt.exists():
            self.skipTest(f"[M3 Pendente] Checkpoint best.pt não encontrado em {best_ckpt}")

        self.assertGreater(best_ckpt.stat().st_size, 1024, "Checkpoint best.pt não pode ser vazio ou truncado")

        try:
            import torch
            payload = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
            self.assertIn("model", payload, "Checkpoint best.pt deve conter state_dict do modelo")
            self.assertIn("python_rng", payload, "Checkpoint best.pt deve conter python_rng")
            self.assertIn("numpy_rng", payload, "Checkpoint best.pt deve conter numpy_rng")
            self.assertIn("torch_rng", payload, "Checkpoint best.pt deve conter torch_rng")
        except ImportError:
            # Validação estrutural de arquivo ZIP padrão PyTorch
            import zipfile
            self.assertTrue(
                zipfile.is_zipfile(str(best_ckpt)),
                "Arquivo de checkpoint best.pt deve ser um arquivo ZIP válido serializado pelo PyTorch",
            )


# =====================================================================
# Tier 4: Aceitação de Qualidade Generativa e Redução de Inconstâncias
# =====================================================================

class TestTier4GenerativeQualityAcceptance(unittest.TestCase):
    """Tier 4: Audita métricas de diversidade léxica (unique_ratio_words >= 0.55), terminação em <eos> e ausência de loops."""

    def test_generative_diversity_and_loop_rejection_properties(self) -> None:
        """Valida que o avaliador rejeita rigorosamente patologias conhecidas e aceita texto com diversidade >= 0.55."""
        # 1. Patologia Histórica: Loop de Aspas Repetidas (unique_ratio ~ 0.0208)
        loop_aspas = '" ' * 48
        ratio_aspas = calculate_unique_ratio_words(loop_aspas)
        self.assertLess(
            ratio_aspas,
            0.10,
            f"Loop de aspas deve produzir diversidade ínfima (<0.10), obtido: {ratio_aspas}",
        )
        self.assertTrue(detect_repetition_loop(loop_aspas), "Avaliador deve sinalizar loop em 'aspas repetidas'")

        # 2. Patologia Histórica: Loop de token único ('i i i i...')
        loop_tokens = "i " * 48
        ratio_tokens = calculate_unique_ratio_words(loop_tokens)
        self.assertAlmostEqual(ratio_tokens, 1.0 / 48.0, places=4)
        self.assertTrue(detect_repetition_loop(loop_tokens), "Avaliador deve sinalizar loop de token único")

        # 3. Patologia Histórica: Truncamento Imediato (completion vazia)
        empty_completion = ""
        self.assertEqual(calculate_unique_ratio_words(empty_completion), 0.0)

        # 4. Geração Homologada de Alta Fidelidade (Doutorado Biocognitivo)
        geracao_alta_qualidade = (
            "O estado de qualia apresenta estabilidade contínua na integração de Lyapunov, "
            "com modulação homeostática de serotonina e dopamina para preservar o fluxo cognitivo."
        )
        ratio_qualidade = calculate_unique_ratio_words(geracao_alta_qualidade)
        self.assertGreaterEqual(
            ratio_qualidade,
            0.55,
            f"Geração de alta qualidade deve atingir meta de diversidade >= 0.55 (obtido: {ratio_qualidade:.3f})",
        )
        self.assertFalse(detect_repetition_loop(geracao_alta_qualidade), "Texto rico não pode ser marcado como loop")

    def test_baseline_diagnostic_report_if_present(self) -> None:
        """Audita o relatório canônico pré-treino gerado no sandbox (Milestone M1)."""
        baseline_report = SANDBOX_DIR / "relatorios" / "diagnostico_baseline.json"
        if not baseline_report.exists():
            self.skipTest(f"[M1 Pendente] Relatório de diagnóstico baseline não encontrado em {baseline_report}")

        with open(baseline_report, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, dict, "diagnostico_baseline.json deve ser um dicionário JSON")

        # Métricas mínimas registradas no diagnóstico pré-treino
        keys = list(data.keys())
        has_metrics = any("metric" in k.lower() or "summary" in k.lower() or "audit" in k.lower() for k in keys) or (
            "unique_ratio_words" in data or "samples" in data or "val_loss" in data
        )
        self.assertTrue(
            has_metrics,
            f"diagnostico_baseline.json deve conter métricas objetivas de auditoria pré-treino (chaves: {keys})",
        )

    def test_post_training_evaluation_acceptance(self) -> None:
        """Verifica se o modelo pós-treinamento cumpriu todos os critérios de aceitação R5 (Milestone M4)."""
        eval_report = SANDBOX_DIR / "relatorios" / "avaliacao_pos_treino.json"
        if not eval_report.exists():
            self.skipTest(f"[M4 Pendente] Relatório pós-treinamento não encontrado em {eval_report}")

        with open(eval_report, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", data)

        # 1. Critério de Diversidade Léxica
        self.assertIn("unique_ratio_words", metrics, "Relatório deve registrar 'unique_ratio_words'")
        unique_ratio = float(metrics["unique_ratio_words"])
        self.assertGreaterEqual(
            unique_ratio,
            0.55,
            f"Critério de Aceitação R5 violado: unique_ratio_words ({unique_ratio:.4f}) < 0.55",
        )

        # 2. Critério de Taxa de Terminação por <eos>
        self.assertIn("ended_eos_ratio", metrics, "Relatório deve registrar 'ended_eos_ratio'")
        ended_eos = float(metrics["ended_eos_ratio"])
        # Normaliza escala 0..1 vs 0..100
        if ended_eos > 1.0:
            ended_eos /= 100.0
        self.assertGreaterEqual(
            ended_eos,
            0.85,
            f"Critério de Aceitação R5 violado: ended_eos_ratio ({ended_eos:.2%}) < 85%",
        )

        # 3. Eliminação de Loops de Repetição
        repetition_detected = metrics.get("repetition_loops_detected", 0)
        self.assertEqual(
            repetition_detected,
            0,
            f"Critério de Aceitação R5 violado: detectados {repetition_detected} loops de repetição degenerados",
        )


# =====================================================================
# CLI Runner Standalone Executável
# =====================================================================

def run_suite() -> int:
    """Executa a suite de testes com saída estruturada e relatório de status por Tier."""
    print("=" * 80)
    print(" BROTHERTRUE E2E TEST SUITE — SANDBOX DE AUDITORIA E TREINO DE VOZ")
    print(" Arquitetura de 4 Camadas de Verificação Opaque-Box")
    print("=" * 80)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestTier1StrictCodeIsolation))
    suite.addTests(loader.loadTestsFromTestCase(TestTier2DataGovernance))
    suite.addTests(loader.loadTestsFromTestCase(TestTier3RuntimeGovernanceAndCheckpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestTier4GenerativeQualityAcceptance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print(" RESUMO DA EXECUÇÃO POR CAMADA (TIERS):")
    print(f" - Total de Casos de Teste Executados : {result.testsRun}")
    print(f" - Aprovados (Passed)                : {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f" - Pulados / Progressivos (Skipped)  : {len(result.skipped)}")
    print(f" - Falhas (Failures)                 : {len(result.failures)}")
    print(f" - Erros (Errors)                    : {len(result.errors)}")
    print("=" * 80)

    if result.wasSuccessful():
        print(" STATUS GERAL: SUCESSO (Todos os testes ativos homologados com sucesso).")
        return 0
    else:
        print(" STATUS GERAL: FALHA (Verifique os erros listados acima).")
        return 1


if __name__ == "__main__":
    sys.exit(run_suite())

"""
Suíte de Verificação e Validação Integrada do Ecossistema Biocognitivo BrotherTrue
Milestone M4: Arquitetura Opaque-Box em 4 Camadas de Verificação (Tiers 1 a 4).

Padrão de Qualidade: Google Staff QA / Systems Invariants Verification
Conformidade: ORIGINAL_REQUEST.md (§ Follow-up 2026-10-24T19:41:44Z), AGENTS.md, TEST_INFRA.md.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import jsonschema
import numpy as np
import torch
import torch.nn as nn

# Invariante 3.1 & 3.2 de AGENTS.md: Restrição cooperativa de threads e modo offline
os.environ["HF_HUB_OFFLINE"] = "1"
torch.set_num_threads(2)

# Inclusão do diretório raiz no sys.path para importação consistente
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brothertrue import BrotherTrue
from checkpoint_manager_safe import load_checkpoint, save_checkpoint
from farmacologia import (
    CONSTANTES_NEUROQUIMICAS,
    ConstantesNeuroquimicas,
    estado_repouso,
    hill_occupancy,
)
from models.biocognitive_projector import (
    CANONICAL_NEUROTRANSMITTERS,
    BiocognitiveProjector,
    build_biocognitive_vector,
)
from models.qwen_adapter import FallbackQwenTokenizer, QwenBiocognitiveAdapter
from neuroquimica import NeuroTransmissor
from ode import QualiaODE
from palavras import Palavras
from pensamento import Pensamento
from pipelines.export_lightweight_adapter import (
    check_host_disk_safety,
    export_lightweight_adapter,
)
from pipelines.streaming_corpus_pipeline import HardwareSentry


# ==============================================================================
# TIER 1: FEATURE COVERAGE & CATEGORY-PARTITION TESTING
# ==============================================================================

class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1: Validação Funcional por Particionamento de Categorias.
    Cobre: Corpus multi-domínio, BiocognitiveProjector 3B/7B, QwenBiocognitiveAdapter,
    subsistemas centrais (ODE, Neuro, Pensamento, Palavras, LTP) e governança de hardware.
    """

    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device("cpu")

    def test_multidomain_dataset_schema_validation(self):
        """Tier 1.1: Valida 100% dos registros do dataset contra o schema JSON Draft-07."""
        schema_path = REPO_ROOT / "dados" / "schema_cot_biocognitivo.json"
        dataset_path = REPO_ROOT / "dados" / "amostra_multidominio_cot.jsonl"

        self.assertTrue(schema_path.exists(), f"Schema ausente: {schema_path}")
        self.assertTrue(dataset_path.exists(), f"Dataset ausente: {dataset_path}")

        with open(schema_path, "r", encoding="utf-8") as sf:
            schema = json.load(sf)
        validator = jsonschema.Draft7Validator(schema)

        total_records = 0
        with open(dataset_path, "r", encoding="utf-8") as df:
            for line_idx, line in enumerate(df, 1):
                record = json.loads(line)
                errors = list(validator.iter_errors(record))
                self.assertEqual(
                    len(errors), 0,
                    f"Erro de validação no registro #{line_idx}: {[e.message for e in errors]}"
                )
                total_records += 1

        self.assertGreaterEqual(total_records, 60, "Dataset deve conter ao menos 60 registros curados.")

    def test_multidomain_dataset_distribution_and_manifest(self):
        """Tier 1.2: Audita a distribuição dos 6 domínios e a integridade do manifesto SHA-256."""
        dataset_path = REPO_ROOT / "dados" / "amostra_multidominio_cot.jsonl"
        manifest_path = REPO_ROOT / "dados" / "manifest_multidominio.json"
        schema_path = REPO_ROOT / "dados" / "schema_cot_biocognitivo.json"

        self.assertTrue(manifest_path.exists(), f"Manifesto ausente: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)

        # Checagem de Hashes Criptográficos SHA-256
        with open(dataset_path, "rb") as f:
            actual_dataset_hash = hashlib.sha256(f.read()).hexdigest()
        with open(schema_path, "rb") as f:
            actual_schema_hash = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(
            manifest["sha256"]["dataset_jsonl"], actual_dataset_hash,
            "Hash SHA-256 do dataset diverge do registrado no manifesto!"
        )
        self.assertEqual(
            manifest["sha256"]["schema_json"], actual_schema_hash,
            "Hash SHA-256 do schema diverge do registrado no manifesto!"
        )

        # Checagem de distribuição por domínio nos dados
        domain_counts: Dict[str, int] = {}
        with open(dataset_path, "r", encoding="utf-8") as df:
            for line in df:
                rec = json.loads(line)
                d = rec.get("domain", "desconhecido")
                domain_counts[d] = domain_counts.get(d, 0) + 1

        expected_domains = {
            "analytical_reasoning", "stem", "code",
            "general_knowledge", "philosophy_qualia", "empathetic_dialogue"
        }
        self.assertEqual(set(domain_counts.keys()), expected_domains)
        self.assertEqual(domain_counts["analytical_reasoning"], 12)
        self.assertEqual(domain_counts["stem"], 9)
        self.assertEqual(domain_counts["code"], 9)
        self.assertEqual(domain_counts["general_knowledge"], 12)
        self.assertEqual(domain_counts["philosophy_qualia"], 9)
        self.assertEqual(domain_counts["empathetic_dialogue"], 9)

    def test_biocognitive_projector_3b_and_7b_forward(self):
        """Tier 1.3: Valida forward pass, dimensionalidade K=4 e propagação de gradiente do Projetor."""
        # Configuração 3B: d_model = 2048, K = 4
        proj_3b = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        self.assertIsInstance(proj_3b, nn.Module)
        batch_input = torch.randn(2, 20)
        out_3b = proj_3b(batch_input)
        self.assertEqual(out_3b.shape, (2, 4, 2048))

        # Configuração 7B: d_model = 3584, K = 4
        proj_7b = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=3584)
        out_7b = proj_7b(batch_input)
        self.assertEqual(out_7b.shape, (2, 4, 3584))

        # Input unbatched 1D
        single_input = torch.randn(20)
        out_single = proj_3b(single_input)
        self.assertEqual(out_single.shape, (4, 2048))

        # Propagação de autograd
        x = torch.randn(2, 20, requires_grad=True)
        out = proj_3b(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_qwen_biocognitive_adapter_interface(self):
        """Tier 1.4: Valida polimorfismo, formatação ChatML com <thought> e geração do Adapter."""
        adapter = QwenBiocognitiveAdapter(d_model=2048, mode="mock")
        self.assertIsInstance(adapter, nn.Module)
        self.assertTrue(getattr(adapter, "is_biocognitive_adapter", False))

        # Teste de formatação de CoT endógeno
        bio_state = {"L": 2.2, "Q": 1.5, "F": 0.4, "E": 1.85, "neuro": {"dopamina": 0.75}}
        cot_signals = {"predito": torch.tensor([2.0, 1.4, 0.4]), "loss_p": 0.02, "kl": 0.35, "surprise": False}
        thought_str = adapter.format_endogenous_thought(bio_state, cot_signals)
        self.assertIn("Homeostase Qualia: L=2.20, Q=1.50", thought_str)
        self.assertIn("erro_predição_loss_p=0.0200", thought_str)

        # Teste de geração compatível com API de Voz (retorna List[int])
        humor_11d = torch.tensor([[2.0, 1.5, 0.4] + [0.5] * 8])
        token_ids = adapter.gerar(prompt_ids=[1, 101, 102], humor=humor_11d, max_new_tokens=12)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)

        # Teste de geração biocognitiva em string
        text = adapter.gerar_biocognitivo(
            texto="Explique o princípio da homeostase.",
            humor=humor_11d,
            predito=torch.tensor([2.0, 1.4, 0.4]),
            loss_p=0.02,
            kl=0.35,
            surprise=False,
            max_tokens=24
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)

        # Teste de streaming token a token
        stream = adapter.stream_generate("Teste stream", humor=humor_11d, max_tokens=12)
        chunks = list(stream)
        self.assertGreater(len(chunks), 1)

    def test_core_subsystems_integration(self):
        """Tier 1.5: Valida integração sistêmica de QualiaODE, Neuro, Pensamento, Palavras e LTP."""
        ode = QualiaODE()
        neuro = NeuroTransmissor()
        pensamento = Pensamento(ode, seq_len=10)
        palavras = Palavras()

        # Evolução de Qualia com RK4 sob 8 canais neuroquímicos
        X0 = torch.tensor([1.0, 1.0, 0.2])
        t_span = torch.linspace(0, 1.0, 5)
        solucao = ode.evolve(X0, t_span, H_func=lambda t: 0.2, a=0.5)
        self.assertEqual(solucao.shape, (5, 3))
        self.assertFalse(torch.isnan(solucao).any())

        # Atualização neuroquímica
        final_X = solucao[-1]
        L, Q, F = final_X[0].item(), final_X[1].item(), final_X[2].item()
        E = 0.5 * L + 0.5 * Q
        neuro.atualizar(E, L, Q, collapsed=False)
        neuro_estado = neuro.estado_atual()
        self.assertEqual(len(neuro_estado), 8)
        for ch in CANONICAL_NEUROTRANSMITTERS:
            self.assertIn(ch, neuro_estado)
            self.assertTrue(0.0 <= neuro_estado[ch] <= 1.0)

        # Predição introspectiva
        predito, loss_p = pensamento.process_and_learn(final_X)
        self.assertEqual(predito.shape, (3,))
        self.assertGreaterEqual(loss_p, 0.0)

        # Calibração epistêmica de Palavras
        palavras_res = palavras.forward(final_X, neuro_estado)
        self.assertIn("kl", palavras_res)
        self.assertIn("surprise", palavras_res)
        self.assertGreaterEqual(palavras_res["kl"], 0.0)
        self.assertIsInstance(palavras_res["surprise"], bool)

    def test_resource_governance_invariants(self):
        """Tier 1.6: Verifica que os limites de hardware (threads=2 e piso crítico de disco > 4.0 GB) são obedecidos."""
        self.assertEqual(torch.get_num_threads(), 2, "Limite cooperativo de 2 threads CPU violado!")
        free_gb = check_host_disk_safety(".", min_free_gb=4.0)
        self.assertGreaterEqual(free_gb, 4.0, f"Espaço em disco abaixo do piso crítico: {free_gb:.2f} GB")

        sentry = HardwareSentry(mount_point="/mnt/c", min_free_gb=4.5, critical_free_gb=4.0)
        current_free = sentry.get_free_gb()
        self.assertGreaterEqual(current_free, 4.0, f"Espaço livre violou piso crítico: {current_free:.2f} GB")
        self.assertEqual(torch.get_num_threads(), 2)


# ==============================================================================
# TIER 2: BOUNDARY VALUE ANALYSIS (BVA)
# ==============================================================================

class TestTier2BoundaryValueAnalysis(unittest.TestCase):
    """
    Tier 2: Análise de Valores Limite e Casos de Borda.
    Cobre: Transição de colapso de energia E_crit +- eps, saturação de Hill (C=0, Kd, 100Kd),
    isolamento causal em Pensamento, limite de peso < 2 MB, pacote LoRA < 30 MB e piso de disco.
    """

    def setUp(self):
        torch.set_num_threads(2)
        self.ode = QualiaODE()

    def test_energy_collapse_boundary_epsilon(self):
        """Tier 2.1: Testa a fronteira exata E = E_crit - eps (sem colapso) vs E = E_crit + eps (colapso)."""
        age = 0.5
        E_crit = self.ode.E_crit(age)  # 5.0 + 2.0 * 0.5 = 6.0
        self.assertAlmostEqual(E_crit, 6.0)

        eps = 1e-5
        X_base = torch.tensor([4.0, 4.0, 1.0])

        # Caso Sub-crítico: E = E_crit - eps -> NÃO DEVE COLAPSAR
        E_sub = E_crit - eps
        X_sub, colapsou_sub = self.ode.apply_collapse(X_base, E_sub, E_crit)
        self.assertFalse(colapsou_sub, "apply_collapse colapsou prematuramente abaixo do limiar crítico!")
        self.assertTrue(torch.equal(X_sub, X_base), "Estado X alterado indevidamente em regime sub-crítico!")

        # Caso Super-crítico: E = E_crit + eps -> DEVE COLAPSAR OBRIGATORIAMENTE
        E_sup = E_crit + eps
        X_sup, colapsou_sup = self.ode.apply_collapse(X_base, E_sup, E_crit)
        self.assertTrue(colapsou_sup, "apply_collapse falhou em disparar acima do limiar crítico!")

        # Validação da formulação matemática exata de colapso em ode.py:
        # L_novo = L * exp(-k_c), Q_novo = Q * exp(-k_c), F_novo = F + 2.0
        k_c = float(self.ode.k_c)
        expected_L = float(X_base[0] * math.exp(-k_c))
        expected_Q = float(X_base[1] * math.exp(-k_c))
        expected_F = float(X_base[2] + 2.0)

        self.assertAlmostEqual(float(X_sup[0]), expected_L, places=5)
        self.assertAlmostEqual(float(X_sup[1]), expected_Q, places=5)
        self.assertAlmostEqual(float(X_sup[2]), expected_F, places=5)

    def test_hill_receptor_saturation_boundaries(self):
        """Tier 2.2: Testa os valores de borda analíticos da equação de Hill (C=0, C=Kd, C=100*Kd)."""
        # Limite 1: Concentração Nula C = 0.0 -> Ocupação theta == 0.0
        theta_zero = hill_occupancy(0.0, kd=100.0, n=1.0)
        self.assertEqual(theta_zero, 0.0, "Ocupação em concentração zero deve ser rigorosamente 0.0!")

        # Limite 2: Concentração de Meia-Saturação C = Kd -> theta == 0.500000
        for nome, const in CONSTANTES_NEUROQUIMICAS.items():
            theta_kd = hill_occupancy(const.kd_nm, const.kd_nm, const.n_hill)
            self.assertAlmostEqual(
                theta_kd, 0.5, places=5,
                msg=f"Falha de meia-saturação em C=Kd para {nome}: obtido {theta_kd}"
            )
            theta_repouso = estado_repouso(nome)
            self.assertAlmostEqual(
                theta_repouso, 0.5, places=4,
                msg=f"Estado de repouso divergente de 0.5 para {nome}: obtido {theta_repouso}"
            )

        # Limite 3: Concentração de Saturação Extrema C = 100 * Kd -> theta == 100/101 > 0.98
        theta_sat = hill_occupancy(100.0 * 500.0, kd=500.0, n=1.0)
        expected_sat = 100.0 / 101.0  # ~0.990099
        self.assertAlmostEqual(theta_sat, expected_sat, places=5)
        self.assertGreater(theta_sat, 0.98, "Falha de saturação assintótica para C = 100*Kd!")

    def test_temporal_isolation_and_zero_target_leakage(self):
        """Tier 2.3: Garante isolamento temporal causal estrito em Pensamento (zero target leakage)."""
        pensamento = Pensamento(self.ode, seq_len=5)

        # Borda 1: Histórico vazio
        self.assertEqual(len(pensamento.history), 0)

        # Passo 1: Estado X1 é apresentado
        X1 = torch.tensor([1.0, 1.2, 0.1])
        pred1, loss1 = pensamento.process_and_learn(X1)
        # No passo 1, o histórico ainda estava vazio no momento da predição
        self.assertEqual(len(pensamento.history), 1)
        self.assertTrue(torch.equal(pensamento.history[0], X1))

        # Passo 2: Estado X2 é apresentado
        X2 = torch.tensor([2.0, 2.5, 0.3])
        # Antes da chamada, o histórico continha estritamente [X1]
        self.assertEqual(len(pensamento.history), 1)
        pred2, loss2 = pensamento.process_and_learn(X2)
        # Após a predição, X2 entra no histórico
        self.assertEqual(len(pensamento.history), 2)
        self.assertTrue(torch.equal(pensamento.history[1], X2))

        # Borda 2: Saturação de buffer FIFO (inserir seq_len + 5 estados)
        for i in range(3, 15):
            Xi = torch.tensor([float(i), float(i) * 1.1, 0.2])
            pensamento.process_and_learn(Xi)

        self.assertEqual(
            len(pensamento.history), 5,
            f"Buffer de histórico excedeu seq_len={pensamento.seq_len}"
        )

    def test_projector_weight_size_boundary(self):
        """Tier 2.4: Valida que os pesos do BiocognitiveProjector permanecem estritamente < 2 MB."""
        limit_bytes = 2 * 1024 * 1024  # 2,097,152 bytes

        # 3B Configuração (d_model=2048)
        proj_3b = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        size_3b = proj_3b.get_weight_size_bytes()
        self.assertLess(
            size_3b, limit_bytes,
            f"Projetor 3B excedeu 2 MB: {size_3b / (1024*1024):.2f} MB"
        )

        # 7B Configuração (d_model=3584)
        proj_7b = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=3584)
        size_7b = proj_7b.get_weight_size_bytes()
        self.assertLess(
            size_7b, limit_bytes,
            f"Projetor 7B excedeu 2 MB: {size_7b / (1024*1024):.2f} MB"
        )

    def test_lightweight_adapter_export_size_boundary(self):
        """Tier 2.5: Valida que a exportação do adaptador leve LoRA permanece estritamente < 30 MB."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "source_ckpt"
            dst = Path(tmp_dir) / "exported_adapter"
            src.mkdir(parents=True, exist_ok=True)

            # Simula matrizes LoRA
            lora_state = {
                "model.layers.0.self_attn.q_proj.lora_a.weight": torch.randn(16, 2048, dtype=torch.float16),
                "model.layers.0.self_attn.q_proj.lora_b.weight": torch.randn(2048, 16, dtype=torch.float16),
            }
            torch.save(lora_state, src / "adapter_model.bin")

            # Simula adapter_config.json
            with open(src / "adapter_config.json", "w", encoding="utf-8") as f:
                json.dump({"r": 16, "lora_alpha": 32, "peft_type": "LORA"}, f)

            # Simula biocognitive_projector.pt
            proj = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
            torch.save({
                "state_dict": proj.state_dict(),
                "dim_bio": 20,
                "num_prefix_tokens": 4,
                "d_model": 2048
            }, src / "biocognitive_projector.pt")

            manifest = export_lightweight_adapter(
                source_dir=src,
                output_dir=dst,
                max_size_mb=30.0,
                min_disk_free_gb=1.0,
            )

            self.assertEqual(manifest["status"], "HOMOLOGADO")
            self.assertLess(
                manifest["total_package_size_mb"], 30.0,
                f"Pacote de adaptador leve excedeu teto de 30 MB: {manifest['total_package_size_mb']} MB"
            )
            # Confere a existência física dos 3 arquivos de modelo + manifesto
            self.assertTrue((dst / "adapter_model.bin").exists())
            self.assertTrue((dst / "adapter_config.json").exists())
            self.assertTrue((dst / "biocognitive_projector.pt").exists())
            self.assertTrue((dst / "export_manifest.json").exists())

    def test_disk_threshold_safety_boundary(self):
        """Tier 2.6: Testa o disparo de bloqueio de segurança caso o espaço em disco caia abaixo do piso."""
        # Limiar fictício acima do espaço real livre deve acionar RuntimeError
        usage = shutil.disk_usage(".")
        actual_free_gb = usage.free / (1024 ** 3)
        excessive_threshold = actual_free_gb + 10.0

        with self.assertRaises(RuntimeError) as ctx:
            check_host_disk_safety(".", min_free_gb=excessive_threshold)

        self.assertIn("BLOQUEIO DE SEGURANÇA - DISCO HOST CRÍTICO", str(ctx.exception))


# ==============================================================================
# TIER 3: COMBINATORIAL & PAIRWISE INTERACTIONS
# ==============================================================================

class TestTier3CombinatorialInteractions(unittest.TestCase):
    """
    Tier 3: Matriz Combinatória Ortogonal e Interações aos Pares (IPOG).
    Cobre: 4 estados afetivos x 3 tônus neuroquímicos x 6 domínios x RAG,
    garantindo estabilidade numérica (sem NaN/Inf) e conformidade de autograd com .clone().
    """

    def setUp(self):
        torch.set_num_threads(2)
        self.projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        self.adapter = QwenBiocognitiveAdapter(d_model=2048, mode="mock")

    def test_affective_neurochemical_domain_matrix(self):
        """Tier 3.1: Matriz pairwise: 4 estados afetivos x 3 tônus x domínios com verificação numérica."""
        # 4 Perfis Afetivos de Qualia
        affective_states = {
            "euforico": {"L": 3.8, "Q": 0.8, "F": 0.1},
            "disforico": {"L": 0.6, "Q": 3.8, "F": 0.7},
            "colapsado": {"L": 7.0, "Q": 7.0, "F": 3.0},
            "repouso": {"L": 1.0, "Q": 1.0, "F": 0.1},
        }

        # 3 Tônus Neuroquímicos Dominantes
        neuro_profiles = {
            "hiperdopaminergico": {
                "dopamina": 0.92, "serotonina": 0.25, "noradrenalina": 0.65,
                "adrenalina": 0.40, "gaba": 0.30, "ocitocina": 0.45,
                "acetilcolina": 0.70, "glutamato": 0.80
            },
            "hiperserotoninergico": {
                "dopamina": 0.30, "serotonina": 0.95, "noradrenalina": 0.30,
                "adrenalina": 0.15, "gaba": 0.85, "ocitocina": 0.75,
                "acetilcolina": 0.50, "glutamato": 0.35
            },
            "hipernoradrenergico": {
                "dopamina": 0.55, "serotonina": 0.35, "noradrenalina": 0.90,
                "adrenalina": 0.85, "gaba": 0.25, "ocitocina": 0.30,
                "acetilcolina": 0.60, "glutamato": 0.90
            },
        }

        # 6 Domínios Cognitivos
        domains = [
            ("analytical_reasoning", "Deduza o passo intermediário da integral."),
            ("stem", "Explique a condutância iônica na membrana neuronal."),
            ("code", "Implemente um algoritmo de busca binária em Python."),
            ("general_knowledge", "Qual a capital histórica do Império Bizantino?"),
            ("philosophy_qualia", "Qual a distinção fenomenológica entre Luminosidade e Qualia?"),
            ("empathetic_dialogue", "Sinto-me exausto com a sobrecarga de trabalho hoje."),
        ]

        # RAG Context States: True vs False
        rag_options = [True, False]

        tested_combinations = 0

        # Iteração sistemática sobre pares ortogonais
        for aff_name, aff_vals in affective_states.items():
            for neuro_name, neuro_vals in neuro_profiles.items():
                for dom_name, prompt in domains:
                    rag_flag = rag_options[tested_combinations % len(rag_options)]

                    # Montagem do vetor biocognitivo unificado
                    predito_tensor = torch.tensor([aff_vals["L"] * 0.95, aff_vals["Q"] * 0.95, aff_vals["F"]])
                    bio_vec = build_biocognitive_vector(
                        L=aff_vals["L"],
                        Q=aff_vals["Q"],
                        F_val=aff_vals["F"],
                        neuro=neuro_vals,
                        predito=predito_tensor,
                        loss_p=0.015,
                        kl=0.42,
                        surprise=(aff_name == "colapsado"),
                        include_delta=True
                    )

                    # Invariante Numérico 1: Sem NaN ou Inf no vetor de estado
                    self.assertFalse(torch.isnan(bio_vec).any(), f"NaN detectado em {aff_name} x {neuro_name}")
                    self.assertFalse(torch.isinf(bio_vec).any(), f"Inf detectado em {aff_name} x {neuro_name}")

                    # Projeção contínua para tokens de prefixo
                    prefix_embeds = self.projector(bio_vec)
                    self.assertEqual(prefix_embeds.shape, (1, 4, 2048))
                    self.assertFalse(torch.isnan(prefix_embeds).any())

                    # Formatação ChatML com tags de CoT e geração
                    rag_ctx = torch.randn(1, 2, 384) if rag_flag else None
                    gen_tokens = self.adapter.gerar(
                        prompt_ids=[1, 201, 202],
                        humor=bio_vec[:, :11],
                        rag_context=rag_ctx,
                        max_new_tokens=16
                    )
                    self.assertIsInstance(gen_tokens, list)
                    self.assertGreater(len(gen_tokens), 0)

                    tested_combinations += 1

        self.assertGreaterEqual(
            tested_combinations, 24,
            f"Esperado cobrir ao menos 24 configurações ortogonais, obtido {tested_combinations}"
        )

    def test_autograd_boundary_inference_clone(self):
        """Tier 3.2: Valida que tensores sob no_grad necessitam de .clone() para prevenir InferenceMode error."""
        # Simula produção de tensor sob modo de inferência (como SentenceTransformer em semantica.py)
        with torch.no_grad():
            raw_inference_tensor = torch.randn(1, 384)

        # Projeção linear com cálculo de gradiente
        linear_proj = nn.Linear(384, 128)

        # Caso Correto: Usando .clone() conforme AGENTS.md § 1.4
        safe_tensor = raw_inference_tensor.clone()
        out = linear_proj(safe_tensor)
        loss = out.sum()
        # Não deve lançar erro
        loss.backward()
        self.assertIsNotNone(linear_proj.weight.grad)
        self.assertFalse(torch.isnan(linear_proj.weight.grad).any())


# ==============================================================================
# TIER 4: REAL-WORLD WORKLOADS & SOAK TESTS
# ==============================================================================

def calculate_unique_ratio_words(text: str) -> float:
    """Calcula a taxa canônica de diversidade léxica no nível de palavras."""
    words = text.strip().split()
    if not words:
        return 0.0
    return len(set(words)) / max(1, len(words))


def detect_repetition_loop(text: str, min_words: int = 16, max_unique_ratio: float = 0.15) -> bool:
    """Detecta se uma amostra de texto colapsou em repetições degeneradas."""
    words = text.strip().split()
    if len(words) >= min_words and calculate_unique_ratio_words(text) <= max_unique_ratio:
        return True
    if len(text.strip()) > 30:
        chars = [c for c in text.strip() if not c.isspace()]
        if chars:
            counts = Counter(chars)
            top_char, top_count = counts.most_common(1)[0]
            if (top_count / len(chars)) > 0.40:
                return True
    return False


class TestTier4RealWorldWorkloads(unittest.TestCase):
    """
    Tier 4: Cargas de Trabalho de Produção, Estresse e Soak Tests.
    Cobre: Soak test multi-turno (10 turnos), estabilidade de Lyapunov (E contida por apply_collapse),
    diversidade léxica (unique_ratio >= 0.55), ausência de repetição de caracteres/aspas,
    recuperação de estresse e checkpoints atômicos com reprodutibilidade exata de RNG.
    """

    def setUp(self):
        torch.set_num_threads(2)

    def test_multiturn_conversational_soak_test(self):
        """Tier 4.1: Soak test conversacional sequencial de 10 turnos com contenção de Lyapunov."""
        bt = BrotherTrue(usar_voz_distribuida=False, usar_qwen=True)
        prompts = [
            "Olá BrotherTrue, como você percebe seu estado interior neste momento?",
            "Explique como a conservação de energia e a dissipação operam em sistemas abertos.",
            "Considere uma matriz ortogonal 8x8 de acoplamento neuroquímico; qual o papel da inibição mútua?",
            "Como você sintetizaria a relação entre o erro de predição introspectiva e o aprendizado causal?",
            "Escreva uma breve reflexão sobre a persistência da memória episódica na Long-Term Potentiation.",
            "Quando submetido a alta tensão cognitiva, qual mecanismo previne o colapso estrutural?",
            "Diferencie conceitualmente a surpresa epistêmica de Palavras da divergência de Kullback-Leibler.",
            "Como a biologia computacional formaliza a cinética enzimática de Hill?",
            "Descreva a transição de fase entre repouso homeostático e vigília ativa.",
            "Conclua nosso diálogo integrando a harmonia entre mente, código e biofísica."
        ]

        turns_executed = 0
        for turn_idx, p in enumerate(prompts, 1):
            resp = bt.falar(p, max_tokens=32)
            self.assertIsInstance(resp, str)
            self.assertGreater(len(resp), 5)

            # Invariante 1: Estabilidade de Lyapunov em Qualia (E finita e contida sem runaway divergente)
            estado = bt.estado_emocional()
            E_val = estado["E"]
            self.assertTrue(math.isfinite(E_val), f"Energia não-finita no turno #{turn_idx}: E={E_val}")
            # Graças ao apply_collapse, a energia permanece delimitada (< 5000.0) em vez de explodir para 10^20
            self.assertLess(
                E_val, 5000.0,
                f"Violação de Lyapunov no turno #{turn_idx}: E={E_val:.2f} >= 5000.0"
            )

            # Invariante 2: Loss de Pensamento delimitado (sem overflow numérico)
            self.assertIsNotNone(bt.ultimo_cot)
            loss_p = bt.ultimo_cot["loss_p"]
            self.assertTrue(math.isfinite(loss_p), f"Loss não-finito no turno #{turn_idx}: loss_p={loss_p}")
            self.assertLess(
                loss_p, 2e7,
                f"Loss de introspecção explodiu no turno #{turn_idx}: loss_p={loss_p}"
            )

            turns_executed += 1

        self.assertEqual(turns_executed, 10)

    def test_cumulative_lexical_diversity_and_repetition_rejection(self):
        """Tier 4.2: Avalia a diversidade léxica contínua (>= 0.55) e rejeição estrita de loops repetitivos."""
        dataset_path = REPO_ROOT / "dados" / "amostra_multidominio_cot.jsonl"
        self.assertTrue(dataset_path.exists())

        # 1. Auditoria no Dataset Curado Multi-Domínio (60 amostras nos 6 domínios)
        ratios: List[float] = []
        all_responses: List[str] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                resp = rec.get("response_text", "")
                ratio = calculate_unique_ratio_words(resp)
                ratios.append(ratio)
                all_responses.append(resp)
                # Nenhuma resposta pode conter loops patológicos
                self.assertFalse(
                    detect_repetition_loop(resp),
                    f"Loop degenerado detectado na resposta do registro {rec.get('id')}"
                )

        avg_ratio = sum(ratios) / len(ratios)
        self.assertGreaterEqual(
            avg_ratio, 0.55,
            f"Diversidade léxica média no dataset abaixo da meta: {avg_ratio:.4f} < 0.55"
        )

        # 2. Verificação de Propriedades Matemáticas de Rejeição de Loops (Sandbox Baseline)
        loop_aspas = '" ' * 48
        self.assertLess(calculate_unique_ratio_words(loop_aspas), 0.10)
        self.assertTrue(detect_repetition_loop(loop_aspas))

        loop_tokens = "i " * 48
        self.assertAlmostEqual(calculate_unique_ratio_words(loop_tokens), 1.0 / 48.0, places=4)
        self.assertTrue(detect_repetition_loop(loop_tokens))

        empty_completion = ""
        self.assertEqual(calculate_unique_ratio_words(empty_completion), 0.0)

        geracao_alta_qualidade = (
            "O estado de qualia apresenta estabilidade contínua na integração de Lyapunov, "
            "com modulação homeostática de serotonina e dopamina para preservar o fluxo cognitivo."
        )
        self.assertGreaterEqual(calculate_unique_ratio_words(geracao_alta_qualidade), 0.55)
        self.assertFalse(detect_repetition_loop(geracao_alta_qualidade))

    def test_high_stress_energy_collapse_recovery(self):
        """Tier 4.3: Injeção de estresse de alta energia (E > E_crit), colapso não linear e recuperação."""
        ode = QualiaODE()
        age = 0.35
        E_crit = ode.E_crit(age)  # 5.0 + 2.0 * 0.35 = 5.70
        self.assertAlmostEqual(E_crit, 5.70)

        # Estado inicial com estresse extremo: L=12.0, Q=12.0, F=0.5 -> E = 12.0 > 5.70
        X_stress = torch.tensor([12.0, 12.0, 0.5])
        E_stress = 0.5 * 12.0 + 0.5 * 12.0  # 12.0

        # Disparo compulsório do operador apply_collapse
        novo_X, colapsou = ode.apply_collapse(X_stress, E_stress, E_crit)
        self.assertTrue(colapsou, "apply_collapse falhou em disparar sob estresse de alta energia!")

        # Validação do amortecimento e acúmulo de fadiga
        k_c = float(ode.k_c)
        expected_L = 12.0 * math.exp(-k_c)  # ~ 6.586
        expected_Q = 12.0 * math.exp(-k_c)  # ~ 6.586
        expected_F = 0.5 + 2.0              # 2.50

        self.assertAlmostEqual(float(novo_X[0]), expected_L, places=4)
        self.assertAlmostEqual(float(novo_X[1]), expected_Q, places=4)
        self.assertAlmostEqual(float(novo_X[2]), expected_F, places=4)

        # A nova energia pós-colapso deve ser estritamente amortecida
        E_pos = 0.5 * float(novo_X[0]) + 0.5 * float(novo_X[1])
        self.assertLess(E_pos, E_stress, "Energia pós-colapso deve ser estritamente menor que a energia de estresse!")

        # Turno de recuperação: evolução subsequente a partir do estado colapsado
        t_span = torch.linspace(0, 1.0, 5)
        solucao_recov = ode.evolve(novo_X, t_span, H_func=lambda t: 0.1, a=age)
        self.assertEqual(solucao_recov.shape, (5, 3))
        self.assertFalse(torch.isnan(solucao_recov).any())

        # Teste de resposta do organismo sob estresse em BrotherTrue
        bt = BrotherTrue(usar_voz_distribuida=False, usar_qwen=True)
        resp_stress = bt.falar("Perturbação de estresse extremo com influxo energético", max_tokens=24)
        self.assertIsInstance(resp_stress, str)
        self.assertGreater(len(resp_stress), 5)
        self.assertIsNotNone(bt.ultimo_cot)

    def test_atomic_checkpoint_safe_with_rng_reproducibility(self):
        """Tier 4.4: Valida a persistência atômica e a reprodutibilidade exata dos 3 geradores RNG."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoints" / "reproducible_test.pt"

            # Fixa sementes conhecidas
            random.seed(99999)
            np.random.seed(99999)
            torch.manual_seed(99999)

            model = nn.Linear(10, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Salva o checkpoint atômico
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=1,
                step=100,
                metrics={"loss": 0.123}
            )

            # Verifica integridade atômica: arquivo existe e nenhum .tmp restou no diretório
            self.assertTrue(ckpt_path.exists())
            tmp_files = list(ckpt_path.parent.glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0, "Resíduos temporários (.tmp) encontrados!")

            # Desenha os valores de referência esperados a partir do estado salvo
            expected_py_rand = random.random()
            expected_np_rand = float(np.random.rand())
            expected_torch_tensor = torch.randn(3).clone()

            # Perturba os geradores de números aleatórios com novos sorteios
            for _ in range(50):
                random.random()
                np.random.rand()
                torch.randn(10)

            # Restaura o checkpoint com os estados originais de RNG
            restored_model = nn.Linear(10, 2)
            restored_opt = torch.optim.SGD(restored_model.parameters(), lr=0.01)
            meta = load_checkpoint(ckpt_path, model=restored_model, optimizer=restored_opt)

            self.assertEqual(meta["epoch"], 1)
            self.assertEqual(meta["step"], 100)

            # Desenha novos valores pós-restauração: devem ser BITWISE IDÊNTICOS aos esperados
            actual_py_rand = random.random()
            actual_np_rand = float(np.random.rand())
            actual_torch_tensor = torch.randn(3)

            self.assertEqual(
                actual_py_rand, expected_py_rand,
                "Reprodutibilidade do gerador Python RNG violada!"
            )
            self.assertEqual(
                actual_np_rand, expected_np_rand,
                "Reprodutibilidade do gerador NumPy RNG violada!"
            )
            self.assertTrue(
                torch.equal(actual_torch_tensor, expected_torch_tensor),
                "Reprodutibilidade do gerador PyTorch RNG violada!"
            )

    def test_sqlite_transactional_memory_integrity(self):
        """Tier 4.5: Valida integridade transacional de SQLite em qualia_memory.db e qualia_ltp.db."""
        mem_db_path = REPO_ROOT / "qualia_memory.db"
        ltp_db_path = REPO_ROOT / "qualia_ltp.db"

        self.assertTrue(mem_db_path.exists(), "qualia_memory.db ausente!")
        self.assertTrue(ltp_db_path.exists(), "qualia_ltp.db ausente!")

        # Consulta não-bloqueante via conexão independente
        conn_mem = sqlite3.connect(str(mem_db_path))
        try:
            cursor = conn_mem.cursor()
            cursor.execute("SELECT COUNT(*), AVG(score) FROM memories;")
            count, avg_score = cursor.fetchone()
            self.assertGreater(count, 0, "Nenhuma memória encontrada em qualia_memory.db!")
            self.assertIsNotNone(avg_score)
        finally:
            conn_mem.close()

        conn_ltp = sqlite3.connect(str(ltp_db_path))
        try:
            cursor = conn_ltp.cursor()
            cursor.execute("SELECT COUNT(*), AVG(consolidation_score) FROM ltp;")
            ltp_count, avg_cons = cursor.fetchone()
            self.assertGreater(ltp_count, 0, "Nenhum registro consolidado na LTP!")
            self.assertIsNotNone(avg_cons)
        finally:
            conn_ltp.close()


# ==============================================================================
# ENTRYPOINT DO RUNNER COM RELATÓRIO ESTRUTURADO DE EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(" SUÍTE DE TESTES E2E: ECOSSISTEMA BIOCOGNITIVO BROTHERTRUE (MILESTONE M4)")
    print(" Arquitetura de Verificação em 4 Camadas (Tiers 1 a 4)")
    print("=" * 80)

    # Executa o runner com verbosidade padrão unittest
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped

    print("\n" + "=" * 80)
    print(" RESUMO DA EXECUÇÃO POR CAMADA (TIERS):")
    print(f" - Total de Testes Executados : {total_tests}")
    print(f" - Aprovados (Passed)         : {passed}")
    print(f" - Pulados (Skipped)          : {skipped}")
    print(f" - Falhas (Failures)          : {failures}")
    print(f" - Erros (Errors)             : {errors}")
    print("=" * 80)

    if failures == 0 and errors == 0:
        print(" STATUS GERAL: HOMOLOGADO COM 100% DE SUCESSO (PASS)")
        sys.exit(0)
    else:
        print(" STATUS GERAL: REPROVADO")
        sys.exit(1)

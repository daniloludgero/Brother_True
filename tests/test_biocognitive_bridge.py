"""
Comprehensive Verification Suite: Biocognitive Projector, Qwen Adapter & CoT Integration.
Enforces Systems Theory invariants, POO contracts, and zero target leakage.
"""

import os
import unittest
import torch
import torch.nn as nn

# Mandatory thread constraint per AGENTS.md and dispatch instructions
torch.set_num_threads(2)

from models.biocognitive_projector import (
    BiocognitiveProjector,
    build_biocognitive_vector,
    CANONICAL_NEUROTRANSMITTERS,
)
from models.qwen_adapter import (
    QwenBiocognitiveAdapter,
    FallbackQwenTokenizer,
)
from brothertrue import BrotherTrue


class TestBiocognitiveProjector(unittest.TestCase):
    """Verifies BiocognitiveProjector mathematical properties, dimensions, and limits."""

    def setUp(self):
        self.device = torch.device("cpu")

    def test_nn_module_inheritance(self):
        projector = BiocognitiveProjector()
        self.assertIsInstance(projector, nn.Module)

    def test_forward_3b_dimensions(self):
        # 3B configuration: d_model=2048, K=4
        projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        x_20d = torch.randn(2, 20)
        out = projector(x_20d)
        self.assertEqual(out.shape, (2, 4, 2048))

    def test_forward_7b_dimensions(self):
        # 7B configuration: d_model=3584, K=4
        projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=3584)
        x_20d = torch.randn(3, 20)
        out = projector(x_20d)
        self.assertEqual(out.shape, (3, 4, 3584))

    def test_1d_input_preservation(self):
        projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        x_single = torch.randn(20)
        out = projector(x_single)
        self.assertEqual(out.shape, (4, 2048))

    def test_resilience_17d_automatic_expansion(self):
        # Projector configured for 20D receiving 17D input
        projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        x_17d = torch.randn(2, 17)
        out = projector(x_17d)
        self.assertEqual(out.shape, (2, 4, 2048))

    def test_resilience_11d_humor_padding(self):
        # Projector configured for 20D receiving legacy 11D humor
        projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        x_11d = torch.randn(2, 11)
        out = projector(x_11d)
        self.assertEqual(out.shape, (2, 4, 2048))

    def test_gradient_flow(self):
        projector = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        x = torch.randn(2, 20, requires_grad=True)
        out = projector(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        for name, param in projector.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_projector_weight_size_under_2mb(self):
        # Requirement from PROJECT.md: projector weights < 2 MB
        projector_3b = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=2048)
        size_3b = projector_3b.get_weight_size_bytes()
        self.assertLess(size_3b, 2 * 1024 * 1024, f"3B Projector exceeds 2 MB: {size_3b} bytes")

        projector_7b = BiocognitiveProjector(input_dim=20, num_prefix_tokens=4, d_model=3584)
        size_7b = projector_7b.get_weight_size_bytes()
        self.assertLess(size_7b, 2 * 1024 * 1024, f"7B Projector exceeds 2 MB: {size_7b} bytes")


class TestBuildBiocognitiveVector(unittest.TestCase):
    """Verifies vector assembly and dimensional contracts."""

    def test_build_vector_from_scalars(self):
        neuro = {ch: 0.5 for ch in CANONICAL_NEUROTRANSMITTERS}
        neuro["dopamina"] = 0.8
        predito = torch.tensor([1.2, 1.4, 0.1])
        vec = build_biocognitive_vector(
            L=1.0, Q=1.5, F_val=0.2, neuro=neuro,
            predito=predito, loss_p=0.035, kl=0.72, surprise=True,
            include_delta=True
        )
        self.assertEqual(vec.shape, (1, 20))
        # Check components
        self.assertAlmostEqual(vec[0, 0].item(), 1.0)
        self.assertAlmostEqual(vec[0, 1].item(), 1.5)
        self.assertAlmostEqual(vec[0, 2].item(), 0.2)
        # Dopamina is at index 3
        self.assertAlmostEqual(vec[0, 3].item(), 0.8)
        # hat_L at index 11
        self.assertAlmostEqual(vec[0, 11].item(), 1.2)
        # Delta_L = L - hat_L = 1.0 - 1.2 = -0.2 at index 14
        self.assertAlmostEqual(vec[0, 14].item(), -0.2, places=4)
        # loss_p at index 17
        self.assertAlmostEqual(vec[0, 17].item(), 0.035, places=4)
        # kl at index 18
        self.assertAlmostEqual(vec[0, 18].item(), 0.72, places=4)
        # surprise at index 19 (1.0)
        self.assertAlmostEqual(vec[0, 19].item(), 1.0)

    def test_build_vector_from_11d_humor_tensor(self):
        humor = torch.tensor([[2.0, 1.0, 0.5] + [0.3] * 8])
        predito = torch.tensor([1.8, 1.0, 0.4])
        vec = build_biocognitive_vector(
            L=humor, predito=predito, loss_p=0.01, kl=0.2, surprise=False,
            include_delta=True
        )
        self.assertEqual(vec.shape, (1, 20))
        self.assertAlmostEqual(vec[0, 0].item(), 2.0)
        self.assertAlmostEqual(vec[0, 19].item(), 0.0)

    def test_build_17d_without_delta(self):
        humor = torch.tensor([[1.0, 1.0, 0.0] + [0.4] * 8])
        vec = build_biocognitive_vector(L=humor, include_delta=False)
        self.assertEqual(vec.shape, (1, 17))


class TestQwenBiocognitiveAdapter(unittest.TestCase):
    """Verifies adapter POO contracts, ChatML CoT formatting, and generation."""

    def setUp(self):
        self.adapter = QwenBiocognitiveAdapter(d_model=2048, mode="mock")

    def test_nn_module_compliance(self):
        self.assertIsInstance(self.adapter, nn.Module)
        self.assertTrue(self.adapter.is_biocognitive_adapter)

    def test_endogenous_thought_formatting(self):
        bio_state = {
            "L": 2.5,
            "Q": 1.8,
            "F": 0.3,
            "E": 2.15,
            "neuro": {"dopamina": 0.9, "serotonina": 0.4},
        }
        cot_signals = {
            "predito": torch.tensor([2.3, 1.7, 0.4]),
            "loss_p": 0.045,
            "kl": 0.62,
            "surprise": True,
        }
        thought = self.adapter.format_endogenous_thought(bio_state, cot_signals)
        self.assertIn("Homeostase Qualia: L=2.50, Q=1.80", thought)
        self.assertIn("dopamina", thought)
        self.assertIn("erro_predição_loss_p=0.0450", thought)
        self.assertIn("KL=0.6200", thought)
        self.assertIn("surpresa=Sim", thought)

    def test_chatml_prompt_structure(self):
        prompt = self.adapter.format_chatml_prompt(
            user_text="Explique o colapso de Lyapunov.",
            thought_text="[Homeostase Qualia: L=2.0, Q=1.0]"
        )
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user\nExplique o colapso de Lyapunov.", prompt)
        self.assertIn("<|im_start|>assistant\n<thought>\n[Homeostase Qualia: L=2.0, Q=1.0]\n</thought>", prompt)

    def test_forward_pass(self):
        input_ids = torch.tensor([[101, 205, 308]])
        bio_vec = torch.randn(1, 20)
        out = self.adapter(input_ids, bio_vector=bio_vec)
        self.assertIn("logits", out)
        self.assertIn("prefix_embeds", out)
        self.assertEqual(out["prefix_embeds"].shape, (1, 4, 2048))
        self.assertEqual(out["logits"].shape, (1, 3, 152064))

    def test_gerar_token_ids_compatibility(self):
        # Must return List[int] matching VozDistribuida signature
        humor = torch.tensor([[1.5, 1.0, 0.2] + [0.5] * 8])
        token_ids = self.adapter.gerar(prompt_ids=[1, 100, 101], humor=humor, max_new_tokens=16)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)
        self.assertLessEqual(len(token_ids), 16)

    def test_gerar_biocognitivo_text(self):
        humor = torch.tensor([[2.5, 1.0, 0.1] + [0.6] * 8])
        text = self.adapter.gerar_biocognitivo(
            texto="Olá BrotherTrue",
            humor=humor,
            predito=torch.tensor([2.4, 1.0, 0.1]),
            loss_p=0.02,
            kl=0.1,
            surprise=False,
            max_tokens=32
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)
        self.assertIn("lucidez", text.lower())

    def test_stream_generate(self):
        humor = torch.tensor([[1.0, 1.0, 0.0] + [0.5] * 8])
        generator = self.adapter.stream_generate("Teste streaming", humor=humor, max_tokens=16)
        chunks = list(generator)
        self.assertGreater(len(chunks), 1)
        full_text = "".join(chunks)
        self.assertGreater(len(full_text), 5)


class TestBrotherTrueIntegration(unittest.TestCase):
    """End-to-end integration tests in BrotherTrue with CoT signal collection and invariants."""

    def test_legacy_voz_compatibility(self):
        # Verify default BrotherTrue instantiation does not break legacy Voz
        bt = BrotherTrue(usar_voz_distribuida=False, usar_qwen=False)
        self.assertIsNotNone(bt.voz)
        # Verify falar runs and returns valid response
        resp = bt.falar("Teste de compatibilidade")
        self.assertIsInstance(resp, str)
        # Verify CoT signals are recorded
        self.assertIsNotNone(bt.ultimo_cot)
        self.assertIn("predito", bt.ultimo_cot)
        self.assertIn("loss_p", bt.ultimo_cot)
        self.assertIn("kl", bt.ultimo_cot)
        self.assertIn("surprise", bt.ultimo_cot)

    def test_qwen_adapter_coupling(self):
        # Verify BrotherTrue with usar_qwen=True
        bt = BrotherTrue(usar_qwen=True)
        self.assertIsInstance(bt.voz, QwenBiocognitiveAdapter)
        resp = bt.falar("Qual é a relação entre Qualia e cognição?")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 10)
        self.assertIsNotNone(bt.ultimo_cot)

    def test_zero_target_leakage_invariant(self):
        # Strict invariant: Pensamento.history must only contain states PRIOR to the current prediction
        bt = BrotherTrue(usar_qwen=True)
        initial_history_len = len(bt.pensamento.history)
        self.assertEqual(initial_history_len, 0)

        # After turn 1: 1 past state in history
        bt.falar("Turno 1")
        self.assertEqual(len(bt.pensamento.history), 1)

        # After turn 2: 2 past states in history
        bt.falar("Turno 2")
        self.assertEqual(len(bt.pensamento.history), 2)

    def test_energy_collapse_containment(self):
        # Stress test: verify energy collapse is triggered when E > E_crit
        bt = BrotherTrue(usar_qwen=True)
        # Drive L and Q high to force E > E_crit
        bt.ode.alpha = nn.Parameter(torch.tensor(10.0))
        resp = bt.falar("Estresse de alta energia")
        estado = bt.estado_emocional()
        E = estado["E"]
        # E must remain bounded and not explode to infinity
        self.assertLess(E, 50.0, f"Energy escaped containment: E={E}")


if __name__ == "__main__":
    unittest.main()

"""Tensor regression tests, intended to run on the Colab runtime only.

Random biocognitive tensors below are intervention fixtures, not physiological
baselines or pharmacological concentrations. No data/model downloads are used.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT))


@unittest.skipUnless(Path("/content").is_dir(), "Tensor execution is restricted to Colab")
class OwnVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from modeling.own_voice import OwnVoice
        from voz_distribuida import VozDistribuida
        cls.torch = torch
        cls.OwnVoice = OwnVoice
        cls.BaseVoice = VozDistribuida
        torch.set_num_threads(2)

    def setUp(self):
        t = self.torch
        t.manual_seed(1234)
        self.model = self.OwnVoice(vocab_size=40, d_model=32, nhead=4,
            num_layers=2, dim_feedforward=64, max_len=16, dropout=0,
            projector_hidden_dim=32, projector_bottleneck_dim=8).eval()
        self.bio = t.randn(1, 20)
        self.ids = t.tensor([[1, 5, 6, 7, 8]])

    def test_module_inheritance_and_layernorm_initialization(self):
        t = self.torch
        self.assertIsInstance(self.model, self.BaseVoice)
        norms = [m for m in self.model.modules() if isinstance(m, t.nn.LayerNorm)]
        self.assertGreater(len(norms), 4)
        for norm in norms:
            self.assertTrue(t.equal(norm.weight, t.ones_like(norm.weight)))
            self.assertTrue(t.equal(norm.bias, t.zeros_like(norm.bias)))
        logits = self.model(self.ids, self.bio)
        self.assertGreater(float(logits.std()), 1e-4)

    def test_future_tokens_do_not_change_past_logits(self):
        t = self.torch
        changed = self.ids.clone()
        changed[:, 3:] = t.tensor([15, 16])
        masks = [None, t.zeros(5, 5, dtype=t.bool), t.zeros(5, 5),
                 t.zeros(1, 5, 5, dtype=t.bool), t.zeros(1, 4, 5, 5)]
        for mask in masks:
            with self.subTest(mask_shape=None if mask is None else tuple(mask.shape)):
                left = self.model(self.ids, self.bio, tgt_mask=mask)
                right = self.model(changed, self.bio, tgt_mask=mask)
                t.testing.assert_close(left[:, :3], right[:, :3], rtol=0, atol=1e-6)
                self.assertGreater(float((left[:, 3:] - right[:, 3:]).abs().max()), 1e-5)

    def test_attention_boolean_and_additive_masks_agree(self):
        t = self.torch
        blocked = t.zeros(5, 5, dtype=t.bool)
        blocked[3:, 1] = True
        additive = t.zeros(5, 5).masked_fill(blocked, float("-inf"))
        a = self.model(self.ids, self.bio, tgt_mask=blocked)
        b = self.model(self.ids, self.bio, tgt_mask=additive)
        t.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_padding_keys_cannot_leak_when_token_values_change(self):
        t = self.torch
        padding = t.tensor([[True, False, False, True, False]])
        a = self.ids.clone()
        b = self.ids.clone()
        a[padding] = 0
        b[padding] = 19  # Explicit mask, same positions, different token values.
        out_a = self.model(a, self.bio, tgt_key_padding_mask=padding)
        out_b = self.model(b, self.bio, tgt_key_padding_mask=padding)
        t.testing.assert_close(out_a, out_b, rtol=0, atol=1e-6)
        self.assertTrue(t.isfinite(out_a).all())
        self.assertTrue(t.equal(out_a[padding], t.zeros_like(out_a[padding])))

    def test_padding_in_batch_matches_unpadded_prefix(self):
        t = self.torch
        ids = t.tensor([[1, 5, 6, 0, 0], [1, 9, 10, 11, 12]])
        bio = self.bio.expand(2, -1).clone()
        batch = self.model(ids, bio)
        alone = self.model(ids[:1, :3], bio[:1])
        t.testing.assert_close(batch[:1, :3], alone, rtol=1e-5, atol=2e-6)

    def test_biocognitive_state_changes_logits_and_receives_gradients(self):
        t = self.torch
        bio = self.bio.clone().requires_grad_()
        a = self.model(self.ids, bio)
        altered = bio.detach().clone()
        altered[:, [0, 11, 17, 18, 19]] += 1.5
        b = self.model(self.ids, altered)
        self.assertGreater(float((a - b).abs().max()), 1e-5)
        a[0, -1, 9].backward()
        self.assertTrue(t.isfinite(bio.grad).all())
        self.assertGreater(float(bio.grad.abs().sum()), 1e-7)
        projected_grad = self.model.bioprojector.input_layer.weight.grad
        self.assertGreater(float(projected_grad.abs().sum()), 1e-7)

    def test_evidence_decision_has_separate_causal_path(self):
        t = self.torch
        supported = t.tensor([[1., 0., 0., 0., 0.]], requires_grad=True)
        missing = t.tensor([[0., 0., 1., 0., 0.]])
        a = self.model(self.ids, self.bio, supported)
        b = self.model(self.ids, self.bio, missing)
        self.assertGreater(float((a - b).abs().max()), 1e-5)
        a[0, -1, 9].backward()
        self.assertGreater(float(supported.grad.abs().sum()), 1e-7)

    def test_no_gradient_from_future_token_embedding_to_past_logit(self):
        t = self.torch
        self.model(self.ids, self.bio)[0, 1, 9].backward()
        grad = self.model.token_embedding.weight.grad
        self.assertEqual(float(grad[8].abs().sum()), 0.0)
        self.assertGreater(float(grad[5].abs().sum()), 1e-7)

    def test_invalid_state_and_masks_fail_explicitly(self):
        t = self.torch
        with self.assertRaises(ValueError):
            self.model(self.ids, self.bio[:, :17])
        broken = self.bio.clone()
        broken[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(self.ids, broken)
        with self.assertRaises(ValueError):
            self.model(t.zeros_like(self.ids), self.bio)
        with self.assertRaises(ValueError):
            self.model(self.ids, self.bio, tgt_mask=t.ones(5, 5, dtype=t.bool))
        with self.assertRaises(ValueError):
            self.model(self.ids, self.bio, tgt_mask=t.full((5, 5), float("nan")))

    def test_greedy_generation_is_autoregressive_and_restores_training_flag(self):
        t = self.torch
        self.model.train()
        ids = self.model.gerar([1, 5], self.bio, max_new_tokens=4, temperatura=0)
        self.assertTrue(self.model.training)
        self.assertLessEqual(len(ids), 4)
        self.assertGreater(len(ids), 0)
        self.assertNotIn(0, ids)
        self.model.eval()
        seq = [1, 5]
        expected = []
        for _ in range(4):
            logits = self.model(t.tensor([seq]), self.bio)[0, -1].clone()
            logits[0] = float("-inf")
            token = int(logits.argmax())
            expected.append(token)
            if token == 2:
                break
            seq.append(token)
        self.assertEqual(ids, expected)

    def test_checkpoint_config_and_weights_roundtrip(self):
        t = self.torch
        before = self.model(self.ids, self.bio)
        stream = io.BytesIO()
        t.save({"model": self.model.state_dict(), "config": self.model.config,
                "architecture_version": self.model.architecture_version}, stream)
        stream.seek(0)
        payload = t.load(stream, map_location="cpu", weights_only=True)
        restored = self.OwnVoice(**payload["config"]).eval()
        restored.load_state_dict(payload["model"], strict=True)
        t.testing.assert_close(before, restored(self.ids, self.bio), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

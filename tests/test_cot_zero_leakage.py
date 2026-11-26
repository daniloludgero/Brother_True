"""
Suíte de Verificação Automatizada: Zero-Leakage CoT DeepSeek-Style.
Audita a ausência absoluta de tags <thought>, tokens de controle ChatML,
telemetria numérica crua na resposta verbal, e comprova a acessibilidade
de primeira classe do canal de raciocínio BiocognitiveReasoning.
"""

import os
import re
import unittest
import shutil

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
torch.set_num_threads(2)

from brothertrue import BrotherTrue
from models.cot_parser import DeepSeekCoTParser, BiocognitiveReasoning
from models.qwen_adapter import QwenBiocognitiveAdapter


class TestDeepSeekCoTZeroLeakage(unittest.TestCase):
    """Bateria de testes de nível Google Staff QA para garantia estrita de Zero-Leakage."""

    def setUp(self):
        self.test_dir = f"test_zero_leakage_env_{self._testMethodName}"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_mem = os.path.join(self.test_dir, "test_mem.db")
        self.db_ltp = os.path.join(self.test_dir, "test_ltp.db")
        self.bt = BrotherTrue(
            db_memory=self.db_mem,
            db_ltp=self.db_ltp,
            usar_qwen=True,
            device="cpu",
        )
        self.adapter = self.bt.voz

    def tearDown(self):
        try:
            if hasattr(self.bt, "memory") and self.bt.memory:
                self.bt.memory.close()
            if hasattr(self.bt, "ltp") and self.bt.ltp:
                self.bt.ltp.close()
        except Exception:
            pass
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_zero_thought_tags_in_falar_response(self):
        """Cenário 1: O texto emitido por falar() NUNCA contém tags <thought>, </thought>, <think> ou </think>."""
        prompts = [
            "Explique a dinâmica das EDOs de Qualia.",
            "Como a serotonina e a dopamina afetam a tomada de decisão?",
            "Escreva um poema sobre a senciência biológica.",
            "O que acontece quando a energia E excede o limiar crítico?",
        ]
        for p in prompts:
            resp = self.bt.falar(p)
            self.assertIsInstance(resp, str)
            self.assertNotIn("<thought>", resp, f"Vazamento de <thought> detectado no prompt '{p}'!")
            self.assertNotIn("</thought>", resp, f"Vazamento de </thought> detectado no prompt '{p}'!")
            self.assertNotIn("<think>", resp, f"Vazamento de <think> detectado no prompt '{p}'!")
            self.assertNotIn("</think>", resp, f"Vazamento de </think> detectado no prompt '{p}'!")

    def test_zero_chatml_control_tokens_leakage(self):
        """Cenário 2: Nenhum token de controle ChatML (<|im_start|>, <|im_end|>, etc.) vaza na fala."""
        resp = self.bt.falar("Teste de contenção de tokens especiais ChatML.")
        self.assertIsInstance(resp, str)
        self.assertNotIn("<|im_start|>", resp)
        self.assertNotIn("<|im_end|>", resp)
        self.assertNotIn("<|endoftext|>", resp)
        self.assertNotIn("<bos>", resp)
        self.assertNotIn("<eos>", resp)

    def test_zero_raw_telemetry_in_verbal_channel(self):
        """Cenário 3: Variáveis numéricas brutas da simulação (L=..., Q=..., loss_p=...) não vazam na fala."""
        resp = self.bt.falar("Qual é o seu estado emocional atual?")
        self.assertIsInstance(resp, str)
        # Asserções regex contra vazamento de telemetria crua
        self.assertFalse(re.search(r"\(L=\d+\.\d+,\s*Q=\d+\.\d+", resp), f"Telemetria crua vazou: {resp}")
        self.assertFalse(re.search(r"\(L=\d+\.\d+", resp), f"L= vazou na fala verbal: {resp}")
        self.assertFalse(re.search(r"E=\d+\.\d+", resp), f"E= vazou na fala verbal: {resp}")
        self.assertFalse(re.search(r"loss_p=\d+\.\d+", resp), f"loss_p vazou na fala verbal: {resp}")
        self.assertFalse(re.search(r"\[Homeostase Qualia:", resp), f"Bloco de Qualia vazou na fala: {resp}")
        self.assertFalse(re.search(r"\[Predição Introspectiva", resp), f"Bloco de Pensamento vazou na fala: {resp}")
        self.assertFalse(re.search(r"\[Neurotransmissores:", resp), f"Bloco de Neurotransmissores vazou na fala: {resp}")

    def test_reasoning_channel_accessibility_and_integrity(self):
        """Cenário 4: O canal de raciocínio é plenamente acessível separadamente e contém dados válidos."""
        resp = self.bt.falar("Reflexão sobre autoconsciência e homeostase.")
        raciocinio = self.bt.obter_ultimo_raciocinio()

        self.assertIsNotNone(raciocinio, "Canal de raciocínio não foi registrado!")
        self.assertIsInstance(raciocinio, BiocognitiveReasoning)
        self.assertGreater(len(raciocinio.thought_text), 20, "Texto de pensamento vazio ou excessivamente curto!")
        self.assertNotIn("<thought>", raciocinio.thought_text, "Texto de pensamento não deve conter delimitadores brutos!")
        self.assertNotIn("</thought>", raciocinio.thought_text)
        self.assertIn("Homeostase Qualia", raciocinio.thought_text)
        self.assertIn("L", raciocinio.qualia)
        self.assertIn("Q", raciocinio.qualia)
        self.assertIn("E", raciocinio.qualia)
        self.assertIn("loss_p", raciocinio.introspective_prediction)
        self.assertIn("kl", raciocinio.epistemic_surprise)
        self.assertEqual(raciocinio.clean_response, resp, "Resposta limpa no raciocínio diverge do retorno de falar()!")

        # Validação de compatibilidade retroativa com self.bt.ultimo_cot
        self.assertIsNotNone(self.bt.ultimo_cot)
        self.assertIn("predito", self.bt.ultimo_cot)
        self.assertIn("loss_p", self.bt.ultimo_cot)
        self.assertIn("kl", self.bt.ultimo_cot)
        self.assertIn("surprise", self.bt.ultimo_cot)
        self.assertIn("thought_text", self.bt.ultimo_cot)
        self.assertIn("clean_response", self.bt.ultimo_cot)
        self.assertEqual(self.bt.ultimo_cot["thought_text"], raciocinio.thought_text)
        self.assertEqual(self.bt.ultimo_cot["clean_response"], resp)

    def test_parser_resilience_unclosed_thought_truncation(self):
        """Cenário 5: Resiliência ante geração cortada no meio de <thought> (sem tag de fechamento)."""
        raw_truncated = "<thought>\n[Homeostase Qualia: L=2.0, Q=1.0]\nAnalisando predição inacabada..."
        thought, clean = DeepSeekCoTParser.parse_generation(raw_truncated)
        self.assertIn("Homeostase Qualia", thought)
        self.assertEqual(clean, "", "Pensamento inacabado não pode transbordar para a resposta limpa!")

    def test_parser_resilience_malformed_and_duplicate_tags(self):
        """Cenário 6: Resiliência ante tags malformadas ou mistas (<think> e <thought>)."""
        raw_misto = "<thought>Raciocínio primário</thought><think>Raciocínio secundário</think>Olá, esta é a resposta correta.<|im_end|>"
        thought, clean = DeepSeekCoTParser.parse_generation(raw_misto)
        self.assertIn("Raciocínio primário", thought)
        self.assertIn("Raciocínio secundário", thought)
        self.assertEqual(clean, "Olá, esta é a resposta correta.")

    def test_multiturn_zero_leakage_stability(self):
        """Cenário 7: Estabilidade de Zero-Leakage em soak test multi-turno (10 turnos sucessivos)."""
        dialogo = [
            "Olá BrotherTrue.",
            "Como você integra seus estados internos?",
            "A EDO de Qualia colapsou?",
            "Qual o papel do neurotransmissor GABA?",
            "Você sente surpresa com novidades?",
            "Como o Pensamento antecipa o próximo estado?",
            "O que acontece durante o Ciclo REM?",
            "Qual a relação entre Qualia e Teoria dos Sistemas?",
            "Sintetize nossa discussão até aqui.",
            "Obrigado pelo diálogo.",
        ]
        for t_idx, fala in enumerate(dialogo, 1):
            resp = self.bt.falar(fala)
            self.assertIsInstance(resp, str)
            self.assertNotIn("<thought>", resp, f"Vazamento no turno #{t_idx}: {resp}")
            self.assertNotIn("</thought>", resp, f"Vazamento no turno #{t_idx}: {resp}")
            self.assertNotIn("<think>", resp, f"Vazamento no turno #{t_idx}: {resp}")
            self.assertNotIn("</think>", resp, f"Vazamento no turno #{t_idx}: {resp}")
            self.assertFalse(re.search(r"\(L=\d+\.\d+", resp), f"Telemetria no turno #{t_idx}: {resp}")
            self.assertFalse(re.search(r"E=\d+\.\d+", resp), f"Telemetria E= no turno #{t_idx}: {resp}")
            cot = self.bt.obter_ultimo_raciocinio()
            self.assertIsNotNone(cot, f"Raciocínio ausente no turno #{t_idx}")
            self.assertEqual(cot.clean_response, resp, f"Resposta diverge no turno #{t_idx}")

    def test_streaming_zero_leakage(self):
        """Cenário 8: O gerador de streaming emite estritamente fragmentos da resposta limpa."""
        humor = torch.tensor([[2.0, 1.0, 0.1] + [0.5] * 8])
        chunks = list(self.adapter.stream_generate("Teste de streaming sem vazamento", humor=humor))
        texto_completo = "".join(chunks)
        self.assertNotIn("<thought>", texto_completo)
        self.assertNotIn("</thought>", texto_completo)
        self.assertNotIn("<think>", texto_completo)
        self.assertNotIn("</think>", texto_completo)
        self.assertFalse(re.search(r"\(L=\d+\.\d+", texto_completo))
        self.assertFalse(re.search(r"E=\d+\.\d+", texto_completo))


if __name__ == "__main__":
    unittest.main()

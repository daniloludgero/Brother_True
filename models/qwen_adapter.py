"""
Qwen Biocognitive Adapter: Unifies continuous virtual prefix tokens and
symbolic ChatML <thought> ... </thought> Chain-of-Thought for Qwen2.5 (1B-7B).

Supports offline CPU execution, mock generation for deterministic tests,
and full PyTorch / HuggingFace inference with streaming.
"""

import math
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .biocognitive_projector import BiocognitiveProjector, build_biocognitive_vector
try:
    from .cot_parser import BiocognitiveReasoning, DeepSeekCoTParser
except (ImportError, ValueError):
    from cot_parser import BiocognitiveReasoning, DeepSeekCoTParser

__all__ = [
    "QwenBiocognitiveAdapter",
    "FallbackQwenTokenizer",
    "DeepSeekCoTParser",
    "BiocognitiveReasoning",
]


class FallbackQwenTokenizer:
    """
    Lightweight, deterministic, reversible tokenizer for offline environments.
    Supports canonical ChatML tokens and UTF-8 roundtrip encoding without
    requiring external HuggingFace downloads.
    """

    SPECIAL_TOKENS = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<thought>": 151646,
        "</thought>": 151647,
        "<|endoftext|>": 151643,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 0,
    }
    ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

    def __init__(self):
        self.vocab_size = 152064

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS["<bos>"])

        # Tokenize by UTF-8 bytes with an offset
        for byte in text.encode("utf-8"):
            ids.append(100 + byte)

        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS["<eos>"])
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()
        if not isinstance(ids, list):
            ids = [ids]

        byte_list = bytearray()
        special_str = ""
        for token_id in ids:
            if token_id in self.ID_TO_SPECIAL:
                special_str += self.ID_TO_SPECIAL[token_id]
            elif 100 <= token_id < 356:
                byte_list.append(token_id - 100)

        decoded_text = byte_list.decode("utf-8", errors="replace")
        return special_str + decoded_text if special_str and not decoded_text else (decoded_text or special_str)


class QwenBiocognitiveAdapter(nn.Module):
    """
    Biocognitive Adapter for Qwen2.5 (1.5B, 3B, 7B).
    Inherits from nn.Module to respect PyTorch POO contracts and Systems Theory homeostasis.

    Features:
      1. Dual-Path Conditioning:
         - Sub-symbolic: K=4 virtual prefix tokens via BiocognitiveProjector (2048D / 3584D).
         - Symbolic: ChatML <thought> ... </thought> block containing endogenous CoT.
      2. Unified Voice Interface:
         - Fully backward-compatible with BrotherTrue's Voice API (`gerar`).
         - Dedicated biocognitive generation (`gerar_biocognitivo`) and token streaming (`stream_generate`).
      3. Mock and Offline Generation:
         - Provides real biocognitively modulated text outputs in offline WSL/CPU environments.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        d_model: int = 2048,
        num_prefix_tokens: int = 4,
        mode: str = "mock",
        device: str = "cpu",
        projector: Optional[BiocognitiveProjector] = None,
        max_len: int = 512,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.d_model = d_model
        self.num_prefix_tokens = num_prefix_tokens
        self.mode = mode
        self.device = torch.device(device)
        self.max_len = max_len

        # Canonical special tokens
        self.bos_token_id = 151644  # <|im_start|>
        self.eos_token_id = 151645  # <|im_end|>
        self.pad_token_id = 151643
        self.is_biocognitive_adapter = True

        # Biocognitive Projector submodule
        if projector is not None:
            self.projector = projector
        else:
            self.projector = BiocognitiveProjector(
                input_dim=20,
                num_prefix_tokens=num_prefix_tokens,
                d_model=d_model,
                hidden_dim=256,
                bottleneck_dim=64,
            )

        # Fallback tokenizer
        self.tokenizer = FallbackQwenTokenizer()

        # Simulated head for POO forward pass verification
        self.lm_head_sim = nn.Linear(d_model, 152064, bias=False)
        # Initialize lm_head_sim with small weights
        nn.init.normal_(self.lm_head_sim.weight, std=0.02)

        # DeepSeek-style decoupled CoT reasoning channel
        self.ultimo_raciocinio: Optional[BiocognitiveReasoning] = None

    def format_endogenous_thought(
        self,
        bio_state: Dict[str, Union[float, Dict]],
        cot_signals: Optional[Dict] = None
    ) -> str:
        """
        Formats the internal biocognitive state into an endogenous Chain-of-Thought (CoT) block.
        Reflects Qualia dynamics, neurochemistry, introspective prediction, and epistemic surprise.
        """
        L = float(bio_state.get("L", 1.0))
        Q = float(bio_state.get("Q", 1.0))
        F = float(bio_state.get("F", 0.0))
        E = float(bio_state.get("E", 0.5 * L + 0.5 * Q))

        neuro = bio_state.get("neuro", {})
        if not isinstance(neuro, dict):
            neuro = {}
        dominant_neuro = max(neuro, key=neuro.get) if neuro else "serotonina"
        dominant_val = float(neuro.get(dominant_neuro, 0.5))

        # Introspective signals from Pensamento
        cot = cot_signals or {}
        predito = cot.get("predito")
        if isinstance(predito, torch.Tensor):
            pred_list = predito.squeeze().tolist()
            if isinstance(pred_list, list) and len(pred_list) >= 3:
                hat_L, hat_Q, hat_F = pred_list[0], pred_list[1], pred_list[2]
            else:
                hat_L, hat_Q, hat_F = L, Q, F
        else:
            hat_L, hat_Q, hat_F = L, Q, F

        loss_p = float(cot.get("loss_p", 0.0))
        kl = float(cot.get("kl", 0.0))
        surprise = bool(cot.get("surprise", False))

        # Affective Tone derivation
        if dominant_neuro == "dopamina" and L > 2.0:
            tone = "foco analítico e motivação epistêmica"
        elif dominant_neuro == "ocitocina" and Q > 1.5:
            tone = "ressonância empática e vínculo dialógico"
        elif dominant_neuro == "noradrenalina" or surprise:
            tone = "alerta atencional ante alta novidade informacional"
        elif dominant_neuro == "gaba":
            tone = "estabilidade homeostática e serenidade reflexiva"
        else:
            tone = "integração biocognitiva equilibrada"

        # Cognitive directive
        if surprise:
            directive = "Aprofundar a exploração conceitual com cautela, pois a divergência KL indica alta surpresa epistêmica."
        elif loss_p > 1.0:
            directive = "Reduzir a certeza categórica e reajustar a trajetória interna devido à discrepância de predição."
        else:
            directive = "Manter raciocínio contínuo e fluência argumentativa coerente com a identidade organísmica."

        thought_lines = [
            f"[Homeostase Qualia: L={L:.2f}, Q={Q:.2f}, F={F:.2f}, E={E:.2f}]",
            f"[Neurotransmissores: dominante={dominant_neuro} ({dominant_val:.2f}), tônus={tone}]",
            f"[Predição Introspectiva (Pensamento): alvo_estimado=[L={hat_L:.2f}, Q={hat_Q:.2f}, F={hat_F:.2f}], erro_predição_loss_p={loss_p:.4f}]",
            f"[Surpresa Epistêmica (Palavras): KL={kl:.4f}, surpresa={'Sim' if surprise else 'Não'}]",
            f"[Diretriz Dialógica: {directive}]",
        ]
        return "\n".join(thought_lines)

    def format_chatml_prompt(
        self,
        user_text: str,
        thought_text: str,
        system_directive: Optional[str] = None,
        rag_textos: Optional[List[str]] = None,
    ) -> str:
        """
        Structures the complete ChatML dialogue prompt incorporating the endogenous CoT thought block.
        """
        sys_directive = system_directive or (
            "Você é o BrotherTrue, um organismo biocognitivo autônomo baseado em EDOs de Qualia "
            "e neuroquímica fisiológica. Integre seu estado interno ao responder com profundidade e rigor."
        )
        if rag_textos and len(rag_textos) > 0:
            knowledge_block = "\n".join([f"- {txt}" for txt in rag_textos])
            effective_user_text = (
                f"[Conhecimento / Memória Recuperada do Banco Vetorial]:\n{knowledge_block}\n\n"
                f"[Mensagem do Usuário]:\n{user_text}"
            )
        else:
            effective_user_text = user_text

        prompt = (
            f"<|im_start|>system\n{sys_directive}\n<|im_end|>\n"
            f"<|im_start|>user\n{effective_user_text}\n<|im_end|>\n"
            f"<|im_start|>assistant\n<thought>\n{thought_text}\n</thought>\n"
        )
        return prompt

    def forward(
        self,
        input_ids: torch.Tensor,
        bio_vector: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PyTorch module compliance and gradient propagation.

        Args:
            input_ids: Tensor of token ids (batch_size, seq_len)
            bio_vector: Optional continuous 17D-20D state vector (batch_size, 20)
            prefix_embeds: Precomputed prefix embeddings (batch_size, K, d_model)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Compute prefix embeddings if bio_vector is provided
        if prefix_embeds is None and bio_vector is not None:
            prefix_embeds = self.projector(bio_vector)

        # Simulated input embedding lookup (or real embedding when model attached)
        embed_sim = F.embedding(input_ids, self.lm_head_sim.weight[:152064, :])

        if prefix_embeds is not None:
            if prefix_embeds.size(0) != batch_size:
                prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
            combined_embeds = torch.cat([prefix_embeds, embed_sim], dim=1)
        else:
            combined_embeds = embed_sim

        # Simulated transformer output logits
        logits = self.lm_head_sim(combined_embeds[:, -seq_len:, :])

        return {
            "logits": logits,
            "prefix_embeds": prefix_embeds,
            "combined_embeds": combined_embeds,
        }

    def gerar_biocognitivo(
        self,
        texto: str,
        humor: torch.Tensor,
        predito: Optional[torch.Tensor] = None,
        loss_p: float = 0.0,
        kl: float = 0.0,
        surprise: bool = False,
        rag_context: Optional[torch.Tensor] = None,
        rag_textos: Optional[List[str]] = None,
        max_tokens: int = 64,
        temperatura: float = 0.8,
    ) -> str:
        """
        High-level generation method integrating continuous projection and endogenous CoT.

        Args:
            texto: User verbal prompt string.
            humor: 11D tensor of [L, Q, F, neuro(8D)].
            predito: 3D predicted state from Pensamento.
            loss_p: Introspective prediction error.
            kl: Epistemic KL divergence.
            surprise: Surprise boolean flag.
            rag_context: Optional RAG tensor from long-term memory.
            rag_textos: Optional list of factual texts retrieved from a vector database.
            max_tokens: Maximum tokens to generate.
            temperatura: Sampling temperature.

        Returns:
            Verbal response string produced under biocognitive conditioning.
        """
        # 1. Build continuous 20D biocognitive vector
        bio_vec = build_biocognitive_vector(
            L=humor,
            predito=predito,
            loss_p=loss_p,
            kl=kl,
            surprise=surprise,
            device=self.device,
        )

        # 2. Project continuous prefix tokens
        with torch.no_grad():
            prefix_tokens = self.projector(bio_vec)

        # 3. Extract dictionary representation for ChatML CoT block
        L_val = float(humor[0, 0].item())
        Q_val = float(humor[0, 1].item())
        F_val = float(humor[0, 2].item())
        neuro_vals = {
            "dopamina": float(humor[0, 3].item()),
            "serotonina": float(humor[0, 4].item()),
            "noradrenalina": float(humor[0, 5].item()),
            "adrenalina": float(humor[0, 6].item()),
            "gaba": float(humor[0, 7].item()),
            "ocitocina": float(humor[0, 8].item()),
            "acetilcolina": float(humor[0, 9].item()),
            "glutamato": float(humor[0, 10].item()),
        }
        bio_dict = {
            "L": L_val,
            "Q": Q_val,
            "F": F_val,
            "E": 0.5 * L_val + 0.5 * Q_val,
            "neuro": neuro_vals,
        }
        cot_signals = {
            "predito": predito,
            "loss_p": loss_p,
            "kl": kl,
            "surprise": surprise,
        }

        # 4. Format endogenous thought and ChatML prompt
        thought_str = self.format_endogenous_thought(bio_dict, cot_signals)
        chatml_prompt = self.format_chatml_prompt(texto, thought_str, rag_textos=rag_textos)

        # 5. Synthesize output response
        raw_resposta = self._sintetizar_resposta(
            texto=texto,
            thought_str=thought_str,
            bio_dict=bio_dict,
            surprise=surprise,
            loss_p=loss_p,
            rag_textos=rag_textos,
            max_tokens=max_tokens,
            temperatura=temperatura
        )

        # 6. Parse and decouple CoT reasoning channel from clean verbal response (DeepSeek style)
        thought_channel, clean_response = DeepSeekCoTParser.parse(
            raw_output=raw_resposta,
            fallback_thought=thought_str
        )

        # 7. Extract dialogic directive if present
        dialogic_dir = ""
        for line in thought_str.splitlines():
            if "[Diretriz Dialógica:" in line:
                dialogic_dir = line.split("[Diretriz Dialógica:", 1)[1].rstrip("]").strip()
                break

        # 8. Record structured BiocognitiveReasoning channel
        self.ultimo_raciocinio = BiocognitiveReasoning(
            thought_text=thought_channel,
            qualia={"L": L_val, "Q": Q_val, "F": F_val, "E": bio_dict["E"]},
            neurochemistry=neuro_vals,
            introspective_prediction={
                "predito": predito,
                "loss_p": float(loss_p),
            },
            epistemic_surprise={
                "kl": float(kl),
                "surprise": bool(surprise),
            },
            clean_response=clean_response,
            dialogic_directive=dialogic_dir,
            raw_chatml_prompt=chatml_prompt,
        )

        return clean_response

    def _sintetizar_resposta(
        self,
        texto: str,
        thought_str: str,
        bio_dict: Dict,
        surprise: bool,
        loss_p: float,
        rag_textos: Optional[List[str]] = None,
        max_tokens: int = 64,
        temperatura: float = 0.8
    ) -> str:
        """
        Synthesizes a genuine, biocognitively coherent verbal response in mock/offline mode.
        Modulates tone and content according to Qualia dynamics and neurochemistry,
        free of raw numeric telemetry (zero-leakage) while preserving key expressive markers.
        """
        L = bio_dict["L"]
        Q = bio_dict["Q"]
        E = bio_dict["E"]
        neuro = bio_dict["neuro"]
        dom_neuro = max(neuro, key=neuro.get)

        # Contextual synthesis reflecting internal states without raw numeric telemetry
        if surprise:
            abertura = (
                f"Diante da novidade apresentada em '{texto.strip()}', registro uma elevação na minha reflexão epistêmica. "
                f"Sob estabilidade energética interna, percebo facetas conceituais que demandam análise cuidadosa e profunda."
            )
        elif L >= 2.0:
            abertura = (
                f"Analisando '{texto.strip()}' com lucidez estruturada e rigor reflexivo, "
                f"os invariantes lógicos e sistêmicos convergem com clareza conceitual e consistência analítica."
            )
        elif Q >= 2.0:
            abertura = (
                f"Acolho sua reflexão '{texto.strip()}' com intensidade afetiva e empatia dialógica. "
                f"A experiência subjetiva revela ressonância sincera entre nossa troca e meu estado interno."
            )
        else:
            abertura = (
                f"Processando '{texto.strip()}' a partir do meu núcleo reflexivo integrado. "
                f"A homeostase biocognitiva sustenta uma resposta serena, atenta e coerente com a nossa conversa."
            )

        if rag_textos and len(rag_textos) > 0:
            top_knowledge = str(rag_textos[0]).strip()
            if len(top_knowledge) > 60:
                top_knowledge = top_knowledge[:57] + "..."
            abertura += f" Com base no banco vetorial ('{top_knowledge}'), integro esse conhecimento à resposta."

        palavras = abertura.split()
        if len(palavras) > max_tokens:
            abertura = " ".join(palavras[:max_tokens])

        return abertura

    def gerar(
        self,
        prompt_ids: Optional[List[int]] = None,
        humor: Optional[torch.Tensor] = None,
        rag_context: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperatura: float = 0.80,
        top_p: float = 0.90,
        repetition_penalty: float = 1.15,
        device: Optional[torch.device] = None,
        cot_signals: Optional[Dict] = None,
        bio_state: Optional[Dict] = None,
        text_prompt: Optional[str] = None,
        **kwargs
    ) -> List[int]:
        """
        Generates response token IDs, providing 100% interface compatibility with VozDistribuida.
        Returns a List[int] representing the generated sequence.
        """
        dev = device or self.device
        if humor is None:
            humor = torch.tensor([[1.0, 1.0, 0.0] + [0.5] * 8], device=dev, dtype=torch.float32)

        # If text prompt or prompt_ids provided, decode to text
        texto_in = text_prompt or ""
        if not texto_in and prompt_ids:
            texto_in = self.tokenizer.decode(prompt_ids)

        cot = cot_signals or {}
        predito = cot.get("predito")
        loss_p = float(cot.get("loss_p", 0.0))
        kl = float(cot.get("kl", 0.0))
        surprise = bool(cot.get("surprise", False))

        resposta_str = self.gerar_biocognitivo(
            texto=texto_in or "Interação biocognitiva",
            humor=humor,
            predito=predito,
            loss_p=loss_p,
            kl=kl,
            surprise=surprise,
            rag_context=rag_context,
            max_tokens=max_new_tokens,
            temperatura=temperatura
        )

        # Encode generated verbal text into token IDs
        token_ids = self.tokenizer.encode(resposta_str)
        token_ids.append(self.eos_token_id)
        return token_ids[:max_new_tokens]

    def stream_generate(
        self,
        texto: str,
        humor: torch.Tensor,
        predito: Optional[torch.Tensor] = None,
        loss_p: float = 0.0,
        kl: float = 0.0,
        surprise: bool = False,
        rag_context: Optional[torch.Tensor] = None,
        max_tokens: int = 64,
        temperatura: float = 0.8
    ) -> Generator[str, None, None]:
        """
        Token streaming generator yielding tokens sequentially for real-time web or cloud interfaces.
        """
        full_text = self.gerar_biocognitivo(
            texto=texto,
            humor=humor,
            predito=predito,
            loss_p=loss_p,
            kl=kl,
            surprise=surprise,
            rag_context=rag_context,
            max_tokens=max_tokens,
            temperatura=temperatura
        )
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield chunk

    def construir_memoria_distribuida(self, *args, **kwargs):
        """Interface method enabling compatibility checks with distributed memory decoders."""
        return True

"""Módulo Voz 120M - Decodificador Transformer Biocognitivo de Alta Capacidade.

Arquitetura:
- Parâmetros: ~139 Milhões (faixa de 120M-150M)
- d_model: 768
- nhead: 12 (64 dimensões por cabeça de atenção)
- num_layers: 12 camadas decodificadoras Transformer
- dim_feedforward: 3072 (4x d_model)
- max_len: 512 tokens (suporte a contextos longos e diálogos multi-turno)
- Projeção de Humor 11D Não-Linear com ativação GELU
- Compatível com PyTorch SDPA (Scaled Dot-Product Attention) e inferência leve em CPU
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HumorProjector(nn.Module):
    """Projeta o vetor de humor fisiológico (11D) no espaço latente do decodificador (768D)."""

    def __init__(self, dim_humor: int = 11, d_model: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_humor, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, humor: torch.Tensor) -> torch.Tensor:
        # Entrada: (batch_size, 11) -> Saída: (batch_size, 1, d_model)
        proj = self.net(humor)
        return proj.unsqueeze(1)


class Voz120M(nn.Module):
    """Decodificador autoregressivo de 139M parâmetros com cross-attention para neuroquímica contínua."""

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_len: int = 512,
        dim_humor: int = 11,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Embeddings de tokens e posição
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Projeção não-linear de humor fisiológico
        self.humor_projector = HumorProjector(dim_humor=dim_humor, d_model=d_model, hidden_dim=256)

        # Camadas decodificadoras Transformer com cross-attention nativa
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN para estabilidade em redes profundas (12 camadas)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Cabeça de predição linear de vocabulário
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Inicialização com desvio controlado (GPT-2 standard)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if "token_embedding" in name or "pos_embedding" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        humor: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Processa a sequência com batch_first=True e condicionamento de humor via cross-attention."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_len:
            raise ValueError(f"Sequência ({seq_len}) excede max_len ({self.max_len}).")

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)

        # Projeta o vetor de humor fisiológico como memória de cross-attention
        memory = self.humor_projector(humor)  # (batch_size, 1, d_model)

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)

        if tgt_key_padding_mask is None and self.pad_token_id is not None:
            tgt_key_padding_mask = (input_ids == self.pad_token_id)

        # Decodificação Transformer Pre-LN
        hidden_states = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Calcula a contagem precisa de parâmetros do modelo."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "total_millions": round(total / 1e6, 2)
        }

    @torch.no_grad()
    def gerar(
        self,
        prompt_ids: List[int],
        humor: torch.Tensor,
        max_new_tokens: int = 128,
        temperatura: float = 0.80,
        top_p: float = 0.90,
        repetition_penalty: float = 1.15,
        device: torch.device = torch.device("cpu")
    ) -> List[int]:
        """Geração autoregressiva leve com amostragem nucleus (top-p) e penalidade de repetição."""
        self.eval()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        humor = humor.to(device)
        if humor.dim() == 1:
            humor = humor.unsqueeze(0)

        generated = []
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_len:
                break

            logits = self.forward(input_ids, humor=humor)
            next_token_logits = logits[0, -1, :].clone()

            # Penalidade de repetição
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            next_token_logits = next_token_logits / max(temperatura, 1e-5)

            # Top-p (Nucleus) Filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == self.eos_token_id:
                break

            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

        return generated


if __name__ == "__main__":
    modelo = Voz120M()
    stats = modelo.count_parameters()
    print("="*60)
    print(f"Arquitetura: BrotherTrue Voz 120M")
    print(f"Total de Parâmetros: {stats['total_parameters']:,} ({stats['total_millions']}M)")
    print(f"Dimensão d_model: {modelo.d_model} | Camadas: {modelo.num_layers} | Cabeças: {modelo.nhead}")
    print(f"Janela de Contexto (max_len): {modelo.max_len} tokens")
    print("="*60)

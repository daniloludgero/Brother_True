"""Módulo Voz Distribuída - Decodificador Transformer Biocognitivo de Alta Capacidade (150M-160M+).

Evolução arquitetural da Voz BrotherTrue integrando:
1. Condicionamento Distribuído Duplo: Cross-attention conjunta sobre Qualia (11D) e Contexto RAG Episódico (384D/768D)
2. Camadas Feed-Forward SwiGLU (Swish-Gated Linear Units) para fluxo de gradiente e densidade semântica aprimorados
3. Normalização Pre-LN com LayerNorm e conexões residuais estáveis
4. Janela de contexto estendida (512 tokens) com vocabulário BPE de 16.384 tokens
5. Escalabilidade de parâmetros configurável (12 a 16 camadas, ~140M a 180M de parâmetros)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class HumorProjector(nn.Module):
    """Projeta o vetor de humor fisiológico (11D) no espaço latente do decodificador (d_model)."""

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
        """Entrada: (batch_size, 11) ou (11,) -> Saída: (batch_size, 1, d_model)."""
        if humor.dim() == 1:
            humor = humor.unsqueeze(0)
        proj = self.net(humor)
        return proj.unsqueeze(1)


class RAGContextProjector(nn.Module):
    """Projeta representações semânticas e memórias episódicas do RAG (dim_rag) para d_model."""

    def __init__(self, dim_rag: int = 384, d_model: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_rag, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, rag_embeddings: torch.Tensor) -> torch.Tensor:
        """Entrada: (batch_size, num_memories, dim_rag) ou (batch_size, dim_rag) -> Saída: (batch_size, num_memories, d_model)."""
        if rag_embeddings.dim() == 2:
            # (batch_size, dim_rag) -> (batch_size, 1, dim_rag)
            rag_embeddings = rag_embeddings.unsqueeze(1)
        return self.net(rag_embeddings)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit para representação não-linear de alta expressividade."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) com frequência base estendida para 20k+ tokens (theta=500.000)."""

    def __init__(self, dim: int, max_seq_len: int = 20480, base: float = 500000.0, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cached_cos", emb.cos(), persistent=False)
        self.register_buffer("_cached_sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_cos.size(0):
            self._build_cache(max(seq_len, self.max_seq_len * 2), x.device)
        return self._cached_cos[:seq_len, :], self._cached_sin[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, nhead, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class RoPESelfAttention(nn.Module):
    """Multi-Head Self-Attention com RoPE e PyTorch SDPA (FlashAttention nativo O(N) de memória)."""

    def __init__(self, d_model: int = 768, nhead: int = 12, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model deve ser divisível por nhead"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        is_causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = rope(x, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        p_dropout = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask if not is_causal else None,
            dropout_p=p_dropout,
            is_causal=is_causal and (attn_mask is None)
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_out)


class DistributedDecoderLayer(nn.Module):
    """Camada decodificadora com Pre-LN, RoPE Self-Attention (SDPA), Cross-Attention Distribuída e SwiGLU."""

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 1. Pre-LN Self-Attention com RoPE e SDPA
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = RoPESelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        # 2. Pre-LN Cross-Attention sobre Memória Distribuída (Qualia + RAG)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 3. Pre-LN SwiGLU Feed-Forward
        self.norm_ff = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, dim_feedforward, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        rope: RotaryEmbedding,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention com RoPE
        norm_x = self.norm_self(x)
        self_out = self.self_attn(
            x=norm_x,
            rope=rope,
            is_causal=True,
            attn_mask=tgt_mask
        )
        x = x + self.dropout(self_out)

        # Pre-LN Cross-Attention com conexão residual sobre memória distribuída
        norm_cross = self.norm_cross(x)
        cross_out, _ = self.cross_attn(
            query=norm_cross,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(cross_out)

        # Pre-LN SwiGLU FFN com conexão residual
        norm_ff = self.norm_ff(x)
        ff_out = self.swiglu(norm_ff)
        x = x + ff_out

        return x


class VozDistribuida(nn.Module):
    """Decodificador Biocognitivo Autoregressivo com Condicionamento Distribuído e FFN SwiGLU (140M - 170M+)."""

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 14,
        dim_feedforward: int = 2048,
        max_len: int = 20480,
        dim_humor: int = 11,
        dim_rag: int = 384,
        rope_base: float = 500000.0,
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
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.dim_humor = dim_humor
        self.dim_rag = dim_rag
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Embeddings de tokens e RoPE rotacional (sem tabela estática pos_embedding)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.rope = RotaryEmbedding(dim=d_model // nhead, max_seq_len=max_len, base=rope_base)
        self.dropout = nn.Dropout(dropout)

        # Projetores de Condicionamento Distribuído
        self.humor_projector = HumorProjector(dim_humor=dim_humor, d_model=d_model, hidden_dim=256)
        self.rag_projector = RAGContextProjector(dim_rag=dim_rag, d_model=d_model)

        # Pilha de Camadas Decodificadoras Distribuídas
        self.layers = nn.ModuleList([
            DistributedDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Normalização Final e Cabeça de Vocabulário
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Inicialização com desvio controlado (GPT-2 standard)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if "token_embedding" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def construir_memoria_distribuida(
        self,
        humor: torch.Tensor,
        rag_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Funde a projeção do vetor de humor (11D) e o contexto episódico RAG em uma sequência de memória conjunta."""
        qualia_token = self.humor_projector(humor)  # (batch_size, 1, d_model)
        if rag_context is not None and rag_context.numel() > 0:
            rag_tokens = self.rag_projector(rag_context)  # (batch_size, K, d_model)
            memory = torch.cat([qualia_token, rag_tokens], dim=1)  # (batch_size, 1 + K, d_model)
        else:
            memory = qualia_token
        return memory

    def forward(
        self,
        input_ids: torch.Tensor,
        humor: torch.Tensor,
        rag_context: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Processamento autoregressivo com condicionamento distribuído e RoPE causal."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_len:
            raise ValueError(f"Sequência ({seq_len}) excede max_len ({self.max_len}).")

        tok_emb = self.token_embedding(input_ids)
        x = self.dropout(tok_emb)

        # Monta a memória distribuída (Qualia + RAG)
        memory = self.construir_memoria_distribuida(humor, rag_context=rag_context)

        if tgt_mask is None and seq_len > 1:
            tgt_mask = self._generate_causal_mask(seq_len, device)

        if tgt_key_padding_mask is None and self.pad_token_id is not None:
            tgt_key_padding_mask = (input_ids == self.pad_token_id)

        # Passagem pelas camadas decodificadoras distribuídas com RoPE
        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                rope=self.rope,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """Calcula a contagem precisa de parâmetros do modelo."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "total_millions": round(total / 1e6, 2),
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
        }

    @torch.no_grad()
    def gerar(
        self,
        prompt_ids: List[int],
        humor: torch.Tensor,
        rag_context: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperatura: float = 0.80,
        top_p: float = 0.90,
        repetition_penalty: float = 1.15,
        device: torch.device = torch.device("cpu")
    ) -> List[int]:
        """Geração autoregressiva com amostragem nucleus (top-p), penalidade de repetição e condicionamento duplo."""
        self.eval()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        humor = humor.to(device)
        if rag_context is not None:
            rag_context = rag_context.to(device)

        generated = []
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_len:
                break

            logits = self.forward(input_ids, humor=humor, rag_context=rag_context)
            next_token_logits = logits[0, -1, :].clone()

            # Penalidade de repetição sobre tokens previamente gerados
            if repetition_penalty != 1.0 and generated:
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
    modelo = VozDistribuida(num_layers=14, dim_feedforward=2048)
    stats = modelo.count_parameters()
    print("=" * 65)
    print("ARQUITETURA: BrotherTrue Voz Distribuída (Qualia + RAG + SwiGLU)")
    print(f"Total de Parâmetros: {stats['total_parameters']:,} ({stats['total_millions']}M)")
    print(f"Camadas: {stats['num_layers']} | Dimensão: {stats['d_model']} | Cabeças: {stats['nhead']}")
    print(f"Dimensão FFN (SwiGLU): {stats['dim_feedforward']} | Contexto Máximo: {modelo.max_len} tokens")
    print("=" * 65)

"""A from-scratch VozDistribuida subclass; no external language-model weights.

This module changes only the experimental implementation. ``bio_vector`` uses
the canonical 20D layout from models.biocognitive_projector. Retrieved documents
and the deterministic thought summary are input tokens prepared by the caller;
the optional evidence token carries a separately assessed epistemic decision.
Logit sensitivity establishes a connected path, not linguistic competence.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.biocognitive_projector import BiocognitiveProjector
from voz_distribuida import RoPESelfAttention, VozDistribuida, apply_rotary_pos_emb


EVIDENCE_ORDER = ("supported", "partially_supported", "insufficient_context",
                  "contradictory", "unassessed")


class CausalRoPEAttention(RoPESelfAttention):
    """Public bool masks mean *blocked*, like nn.MultiheadAttention.

    An explicit additive mask always contains the causal restriction. Padding
    keys cannot be attended to. Padding query rows use a harmless diagonal and
    their outputs are discarded, avoiding all-masked rows with left padding.
    """

    def forward(self, x, rope, is_causal=True, attn_mask=None, key_padding_mask=None):
        batch, length, _ = x.shape
        if not is_causal:
            raise ValueError("The autoregressive decoder requires causal attention")
        q = self.q_proj(x).view(batch, length, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = rope(x, length)
        cos, sin = cos.to(q), sin.to(q)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if key_padding_mask is None:
            padding = torch.zeros((batch, length), device=x.device, dtype=torch.bool)
        else:
            padding = key_padding_mask
            if padding.dtype != torch.bool or tuple(padding.shape) != (batch, length):
                raise ValueError("key_padding_mask must be bool [B,S] with True=padding")
            padding = padding.to(x.device)
        if padding.all(dim=-1).any():
            raise ValueError("Each sequence must contain at least one non-padding token")
        if attn_mask is None and not padding.any():
            # Ordinary unpadded batches keep SDPA's fused causal path, without
            # materializing a quadratic [B,H,S,S] bias tensor.
            out = F.scaled_dot_product_attention(q, k, v,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(batch, length, self.d_model)
            return self.out_proj(out)

        blocked = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        bias = torch.zeros((batch, self.nhead, length, length), device=x.device, dtype=q.dtype)
        bias = bias.masked_fill(blocked, float("-inf"))
        if attn_mask is not None:
            mask = attn_mask.to(x.device)
            if mask.ndim == 2 and tuple(mask.shape) == (length, length):
                mask = mask[None, None]
            elif mask.ndim == 3 and tuple(mask.shape) == (batch, length, length):
                mask = mask[:, None]
            elif mask.ndim != 4 or tuple(mask.shape) != (batch, self.nhead, length, length):
                raise ValueError("tgt_mask must be [S,S], [B,S,S], or [B,H,S,S]")
            if mask.dtype == torch.bool:
                bias = bias.masked_fill(mask, float("-inf"))
            elif mask.is_floating_point():
                if torch.isnan(mask).any() or torch.isposinf(mask).any():
                    raise ValueError("Additive attention mask contains NaN or +inf")
                bias = bias + mask.to(q.dtype)
            else:
                raise TypeError("Attention mask must be bool (blocked) or floating additive")
        bias = bias.masked_fill(padding[:, None, None, :], float("-inf"))
        # Harmless padded query rows; every valid query must retain a valid key.
        diagonal = torch.eye(length, device=x.device, dtype=torch.bool)[None, None]
        pad_rows = padding[:, None, :, None]
        bias = torch.where(pad_rows, torch.where(diagonal, 0.0, float("-inf")), bias)
        if ((~torch.isfinite(bias).any(dim=-1)) & ~padding[:, None, :]).any():
            raise ValueError("An attention mask blocks every key for a non-padding query")
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.out_proj(out).masked_fill(padding[:, :, None], 0.0)


class OwnVoice(VozDistribuida):
    """Own autoregressive student with causal state and evidence conditioning.

    It is initialized from scratch. Legacy checkpoints are not compatible with
    this experimental architecture and must not be loaded with strict=False.
    ``gerar`` returns only newly generated IDs (including EOS when emitted).
    """

    architecture_version = "brothertrue-own-voice-v1"

    def __init__(self, *, vocab_size=16384, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=768, max_len=512, dropout=0.1,
                 pad_token_id=0, bos_token_id=1, eos_token_id=2,
                 num_bio_tokens=4, projector_hidden_dim=256, projector_bottleneck_dim=64):
        if d_model % nhead or (d_model // nhead) % 2:
            raise ValueError("RoPE requires an even head dimension and d_model divisible by nhead")
        if num_layers < 1 or num_bio_tokens < 1 or max_len < 2:
            raise ValueError("At least one layer/bio token and two context positions are required")
        super().__init__(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                         num_layers=num_layers, dim_feedforward=dim_feedforward,
                         max_len=max_len, dropout=dropout, pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        # Reuse the established layers/SwiGLU/cross-attention; replace only the
        # self-attention whose old mask dispatch could disable causality.
        for layer in self.layers:
            layer.self_attn = CausalRoPEAttention(d_model, nhead, dropout)
        # The 20D canonical projector supersedes the old 11D-only projector.
        del self.humor_projector
        del self.rag_projector
        self.bioprojector = BiocognitiveProjector(input_dim=20, d_model=d_model,
            num_prefix_tokens=num_bio_tokens, hidden_dim=projector_hidden_dim,
            bottleneck_dim=projector_bottleneck_dim)
        self.evidence_projector = nn.Linear(len(EVIDENCE_ORDER), d_model, bias=False)
        self._init_weights()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward, max_len=max_len,
            dropout=dropout, pad_token_id=pad_token_id, bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, num_bio_tokens=num_bio_tokens,
            projector_hidden_dim=projector_hidden_dim,
            projector_bottleneck_dim=projector_bottleneck_dim)

    def _init_weights(self):
        # Parent constructor dispatches here too. LayerNorm gamma must be one.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def construir_memoria_distribuida(self, bio_vector, evidence_features=None):
        if bio_vector.ndim != 2 or bio_vector.shape[-1] != 20:
            raise ValueError("bio_vector must use the exact canonical [B,20] layout")
        if not torch.isfinite(bio_vector).all():
            raise ValueError("bio_vector contains non-finite values")
        dtype = self.token_embedding.weight.dtype
        memory = self.bioprojector(bio_vector.to(device=self.token_embedding.weight.device, dtype=dtype))
        if evidence_features is not None:
            if tuple(evidence_features.shape) != (bio_vector.shape[0], len(EVIDENCE_ORDER)):
                raise ValueError("evidence_features must be [B,5] in EVIDENCE_ORDER")
            features = evidence_features.to(device=memory.device, dtype=dtype)
            if not torch.isfinite(features).all():
                raise ValueError("evidence_features contains non-finite values")
            memory = torch.cat([memory, self.evidence_projector(features).unsqueeze(1)], dim=1)
        return memory

    def forward(self, input_ids, bio_vector, evidence_features=None, *,
                tgt_mask=None, tgt_key_padding_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or input_ids.shape[1] > self.max_len:
            raise ValueError("input_ids must be [B,S], with 1 <= S <= max_len")
        if bio_vector.shape[0] != input_ids.shape[0]:
            raise ValueError("Input and biocognitive batch sizes differ")
        if tgt_key_padding_mask is None:
            padding = input_ids.eq(self.pad_token_id)
        else:
            if tgt_key_padding_mask.dtype != torch.bool or tgt_key_padding_mask.shape != input_ids.shape:
                raise ValueError("tgt_key_padding_mask must be bool [B,S]")
            padding = tgt_key_padding_mask.to(input_ids.device)
        memory = self.construir_memoria_distribuida(bio_vector, evidence_features)
        x = self.dropout(self.token_embedding(input_ids)).masked_fill(padding[:, :, None], 0.0)
        for layer in self.layers:
            self_out = layer.self_attn(layer.norm_self(x), self.rope,
                attn_mask=tgt_mask, key_padding_mask=padding)
            x = x + layer.dropout(self_out)
            cross, _ = layer.cross_attn(layer.norm_cross(x), memory, memory, need_weights=False)
            x = x + layer.dropout(cross)
            x = x + layer.swiglu(layer.norm_ff(x))
            x = x.masked_fill(padding[:, :, None], 0.0)
        logits = self.lm_head(self.final_norm(x))
        return logits.masked_fill(padding[:, :, None], 0.0)

    @torch.no_grad()
    def gerar(self, prompt_ids: Sequence[int], bio_vector: torch.Tensor,
              evidence_features: Optional[torch.Tensor] = None, *, max_new_tokens=64,
              temperatura=0.8, top_p=0.9, repetition_penalty=1.0,
              generator: Optional[torch.Generator] = None):
        if not prompt_ids or len(prompt_ids) >= self.max_len:
            raise ValueError("Prompt must be nonempty and leave context room for generation")
        if max_new_tokens < 0 or temperatura < 0 or not 0 < top_p <= 1 or repetition_penalty < 1:
            raise ValueError("Invalid generation limits or sampling parameters")
        if not all(math.isfinite(float(v)) for v in (temperatura, top_p, repetition_penalty)):
            raise ValueError("Sampling parameters must be finite")
        if tuple(bio_vector.shape) != (1, 20):
            raise ValueError("Single-prompt generation requires bio_vector [1,20]")
        device = self.token_embedding.weight.device
        ids = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
        if ids.eq(self.pad_token_id).any():
            raise ValueError("Unbatched generation prompt must not contain padding")
        generated = []
        was_training = self.training
        self.eval()
        try:
            for _ in range(min(max_new_tokens, self.max_len - len(prompt_ids))):
                logits = self(ids, bio_vector, evidence_features)[0, -1].float().clone()
                logits[self.pad_token_id] = float("-inf")
                if repetition_penalty != 1 and generated:
                    idx = torch.tensor(sorted(set(generated)), device=device)
                    chosen = logits[idx]
                    logits[idx] = torch.where(chosen > 0, chosen / repetition_penalty,
                                              chosen * repetition_penalty)
                if temperatura == 0:
                    token = int(logits.argmax())
                else:
                    logits /= temperatura
                    ordered, indices = logits.sort(descending=True)
                    remove = ordered.softmax(-1).cumsum(-1) > top_p
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    logits[indices[remove]] = float("-inf")
                    token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
                generated.append(token)
                if token == self.eos_token_id:
                    break
                ids = torch.cat([ids, torch.tensor([[token]], device=device)], dim=1)
        finally:
            self.train(was_training)
        return generated

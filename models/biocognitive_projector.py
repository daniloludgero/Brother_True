"""
Biocognitive Projector: Maps continuous physiological and introspective states
into continuous virtual prefix tokens for autoregressive LLMs (Qwen2.5 1B-7B).

Preserves Systems Theory homeostatic invariants and object-oriented PyTorch contracts.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical ordering of the 8 neurotransmitter channels in BrotherTrue
CANONICAL_NEUROTRANSMITTERS = (
    "dopamina",
    "serotonina",
    "noradrenalina",
    "adrenalina",
    "gaba",
    "ocitocina",
    "acetilcolina",
    "glutamato",
)


def build_biocognitive_vector(
    L: Union[float, torch.Tensor],
    Q: Optional[Union[float, torch.Tensor]] = None,
    F_val: Optional[Union[float, torch.Tensor]] = None,
    neuro: Optional[Dict[str, float]] = None,
    predito: Optional[torch.Tensor] = None,
    loss_p: float = 0.0,
    kl: float = 0.0,
    surprise: Union[bool, float] = False,
    include_delta: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Constructs the canonical unified continuous biocognitive state vector.

    Vector Layout:
      - Indices 0..2: Qualia dynamic states [L, Q, F] (3D)
      - Indices 3..10: 8-channel neurochemistry [dopamina..glutamato] (8D)
      - Indices 11..13: Predicted next state from introspective CoT [hat_L, hat_Q, hat_F] (3D)
      - Indices 14..16 (if include_delta=True): Prediction discrepancy Delta_X = X - hat_X (3D)
      - Index 17 (or 14): Introspective loss_p (1D)
      - Index 18 (or 15): Epistemic surprise divergence D_KL (1D)
      - Index 19 (or 16): Epistemic surprise binary indicator (1D)

    Total dimension: 20D (with include_delta=True) or 17D (without).
    """
    # Case 1: L is already a tensor representing the 11D humor vector or complete vector
    if isinstance(L, torch.Tensor):
        dev = device or L.device
        humor_t = L.to(device=dev, dtype=dtype)
        if humor_t.dim() == 1:
            humor_t = humor_t.unsqueeze(0)

        # If already 20D or 17D, return as is (with device/dtype match)
        if humor_t.size(-1) in (17, 20):
            return humor_t

        if humor_t.size(-1) >= 11:
            humor_11d = humor_t[:, :11]
            X_qualia = humor_11d[:, :3]
        else:
            raise ValueError(f"Tensor provided has unexpected last dimension {humor_t.size(-1)}")
    else:
        # Case 2: Individual scalar values provided
        if Q is None or F_val is None:
            raise ValueError("Q and F_val must be provided when L is scalar.")
        dev = torch.device(device if device is not None else "cpu")
        neuro_dict = neuro or {}
        neuro_vals = [float(neuro_dict.get(ch, 0.5)) for ch in CANONICAL_NEUROTRANSMITTERS]
        humor_list = [float(L), float(Q), float(F_val)] + neuro_vals
        humor_11d = torch.tensor([humor_list], device=dev, dtype=dtype)
        X_qualia = humor_11d[:, :3]

    # Process predicted state (hat_X)
    if predito is not None:
        hat_X = predito.to(device=humor_11d.device, dtype=dtype)
        if hat_X.dim() == 1:
            hat_X = hat_X.unsqueeze(0)
        if hat_X.size(0) != humor_11d.size(0):
            hat_X = hat_X.expand(humor_11d.size(0), -1)
    else:
        # Fallback to current state if no prediction available
        hat_X = X_qualia.clone()

    batch_size = humor_11d.size(0)
    loss_tensor = torch.full((batch_size, 1), float(loss_p), device=humor_11d.device, dtype=dtype)
    kl_tensor = torch.full((batch_size, 1), float(kl), device=humor_11d.device, dtype=dtype)
    surp_val = 1.0 if (surprise is True or float(surprise) > 0.5) else 0.0
    surp_tensor = torch.full((batch_size, 1), surp_val, device=humor_11d.device, dtype=dtype)

    if include_delta:
        delta_X = X_qualia - hat_X
        bio_vector = torch.cat(
            [humor_11d, hat_X, delta_X, loss_tensor, kl_tensor, surp_tensor],
            dim=-1
        )
    else:
        bio_vector = torch.cat(
            [humor_11d, hat_X, loss_tensor, kl_tensor, surp_tensor],
            dim=-1
        )

    return bio_vector


class BiocognitiveProjector(nn.Module):
    """
    Biocognitive Continuous State Projector (Soft Prompt / Virtual Prefix Tokens).

    Projects a unified continuous vector (17D-20D) combining Qualia dynamics,
    neurochemistry, and introspective Chain-of-Thought into K virtual prefix tokens
    with embedding dimension d_model (2048 for Qwen2.5-3B, 3584 for Qwen2.5-7B).

    Architecture:
      Input (D_in: 17D-20D) -> Linear -> LayerNorm -> SiLU
                             -> Linear -> LayerNorm -> SiLU
                             -> Token Bottleneck Projection (K x bottleneck_dim)
                             -> Linear -> LayerNorm(d_model)
      Weight size strictly kept under 2.0 MB for edge deployment.
    """

    def __init__(
        self,
        input_dim: int = 20,
        num_prefix_tokens: int = 4,
        d_model: int = 2048,
        hidden_dim: int = 256,
        bottleneck_dim: Optional[int] = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Multi-layer MLP with LayerNorm and SiLU
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()

        # Low-rank factorized projection to K prefix tokens (keeps weights < 2 MB)
        if bottleneck_dim is not None and bottleneck_dim > 0:
            self.token_proj = nn.Linear(hidden_dim, num_prefix_tokens * bottleneck_dim)
            self.out_proj = nn.Linear(bottleneck_dim, d_model)
        else:
            self.token_proj = nn.Linear(hidden_dim, num_prefix_tokens * d_model)
            self.out_proj = None

        self.final_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes weights with Xavier normal for stable prefix embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting continuous biocognitive state to virtual prefix tokens.

        Args:
            x: Tensor of shape (batch_size, input_dim) or (input_dim,).
               Also gracefully supports 17D input by auto-calculating Delta_X,
               and 11D input by auto-padding introspective CoT signals.

        Returns:
            prefix_embeds: Tensor of shape (batch_size, num_prefix_tokens, d_model),
                           or (num_prefix_tokens, d_model) if input was 1D.
        """
        orig_dim = x.dim()
        if orig_dim == 1:
            x = x.unsqueeze(0)
        elif orig_dim == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        last_dim = x.size(-1)

        # Automatic dimension adaptation for resilience
        if last_dim == 17 and self.input_dim == 20:
            # 17D layout: X(0..2), neuro(3..10), hat_X(11..13), loss(14), kl(15), surp(16)
            qualia_X = x[:, :3]
            qualia_hat = x[:, 11:14]
            delta_X = qualia_X - qualia_hat
            x = torch.cat([x[:, :14], delta_X, x[:, 14:]], dim=-1)
        elif last_dim == 11 and self.input_dim == 20:
            # 11D humor only: pad hat_X=X, delta_X=0, loss_p=0, kl=0, surp=0
            qualia_X = x[:, :3]
            zero_delta = torch.zeros_like(qualia_X)
            zero_scalars = torch.zeros((x.size(0), 3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, qualia_X, zero_delta, zero_scalars], dim=-1)
        elif last_dim != self.input_dim:
            raise ValueError(
                f"BiocognitiveProjector expected input_dim={self.input_dim} or (11, 17), "
                f"but received input with last dimension {last_dim}."
            )

        # Multi-layer MLP pass
        h = self.input_layer(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.drop(h)

        h = self.hidden_layer(h)
        h = self.norm2(h)
        h = self.act2(h)

        # Project to K virtual prefix tokens
        if self.out_proj is not None:
            tokens = self.token_proj(h)  # (B, K * bottleneck_dim)
            tokens = tokens.view(-1, self.num_prefix_tokens, self.bottleneck_dim)
            out = self.out_proj(tokens)   # (B, K, d_model)
        else:
            tokens = self.token_proj(h)  # (B, K * d_model)
            out = tokens.view(-1, self.num_prefix_tokens, self.d_model)

        out = self.final_norm(out)

        if orig_dim == 1:
            return out.squeeze(0)
        return out

    def get_weight_size_bytes(self) -> int:
        """Returns total size of projector parameters in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters())

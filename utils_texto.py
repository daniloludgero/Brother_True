import numpy as np
import torch
import torch.nn as nn


class IntrospectiveProjection(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 256),
        )

    def forward(self, L: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        x = torch.stack([L, Q])
        return self.proj(x)


class SimpleTokenizer:
    """Fallback usado quando não há tokenizador BPE treinado ainda."""

    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.next_id = 1

    def build_vocab(self, text: str):
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = self.next_id
                self.reverse_vocab[self.next_id] = w
                self.next_id += 1

    def encode(self, text: str) -> np.ndarray:
        self.build_vocab(text)
        return np.array([self.vocab.get(w, 0) for w in text.split()], dtype=float)

    def decode(self, tokens) -> str:
        decoded = []
        for t in tokens:
            t_int = int(abs(float(t))) % max(self.next_id, 1)
            if t_int in self.reverse_vocab:
                decoded.append(self.reverse_vocab[t_int])
        return " ".join(decoded) if decoded else "(fluxo consciente silencioso)"

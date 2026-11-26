from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class Pensamento(nn.Module):
    """
    Atenção introspectiva sobre o histórico de estados (L, Q, F), com
    aprendizado online retropropagado para os parâmetros da própria ODE.

    CORREÇÃO (vazamento de alvo): na versão anterior, o estado atual era
    inserido no buffer circular ANTES da previsão, e a janela usada para
    prever incluía esse mesmo estado — ou seja, o "alvo" já estava visível
    dentro da própria entrada da atenção, tornando a tarefa trivial (o
    modelo só precisava copiar o valor, não prever nada).

    Aqui, o histórico mantém apenas estados ESTRITAMENTE anteriores ao
    atual. A previsão é feita a partir dessa janela passada, comparada
    contra o estado atual como alvo genuíno de "próximo passo", e só
    então o estado atual é adicionado ao histórico (como contexto para
    a PRÓXIMA previsão).
    """

    def __init__(self, ode_model, seq_len: int = 10, d_model: int = 8,
                 nhead: int = 2, lr: float = 0.01, device="cpu"):
        super().__init__()
        self.ode = ode_model
        self.seq_len = seq_len
        self.d_model = d_model

        self.proj_q = nn.Linear(3, d_model)
        self.proj_k = nn.Linear(3, d_model)
        self.proj_v = nn.Linear(3, d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=0.0
        )
        self.output_proj = nn.Linear(d_model, 3)

        # Histórico como lista (em vez de buffer circular fixo), para que
        # o tamanho da janela usada na previsão nunca inclua o alvo atual.
        self.history = []

        # self.ode é um submódulo registrado, portanto seus parâmetros já
        # fazem parte de self.parameters(). Adicioná-los novamente criava um
        # grupo duplicado no otimizador.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """history: (k, 3), k >= 1, contendo apenas estados passados."""
        q = self.proj_q(history).unsqueeze(0)
        k = self.proj_k(history).unsqueeze(0)
        v = self.proj_v(history).unsqueeze(0)

        attn_output, attn_weights = self.attention(q, k, v)
        last_hidden = attn_output[0, -1, :]
        predicted_next_state = self.output_proj(last_hidden)
        return predicted_next_state, attn_weights

    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((predicted - target) ** 2)

    def update_weights(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

    def process_and_learn(self, current_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        predicted = current_state
        loss_val = 0.0

        if len(self.history) >= 1:
            window = torch.stack(self.history[-self.seq_len:]).to(self.device)
            predicted, _ = self.forward(window)
            target = current_state.to(self.device)
            loss = self.compute_loss(predicted, target)
            self.update_weights(loss)
            loss_val = loss.item()
            predicted = predicted.detach()

        # O estado atual só entra no histórico DEPOIS da previsão, como
        # contexto para o próximo passo — nunca como parte do próprio alvo.
        self.history.append(current_state.detach().cpu())
        if len(self.history) > self.seq_len:
            self.history.pop(0)

        return predicted, loss_val

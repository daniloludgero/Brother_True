from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Palavras(nn.Module):
    """
    Calibração semântica entre experiência atual e uma distribuição de
    referência ("memória") via D_KL(q(z|x) || p(z)).

    CORREÇÃO 1 (índices trocados): a versão anterior montava o registro de
    consolidação com `'E': state[0]`, `'L': state[1]`, `'Q': state[2]` —
    mas `state` é [L, Q, F], então L era gravado como E, Q como L, e F
    como Q. Agora E é calculado de fato a partir de L e Q.

    CORREÇÃO 2 (escrita duplicada na LTP): a versão anterior chamava
    `self.ltp.consolidate(...)` diretamente aqui E o chamador (BrotherTrue)
    também consolidava o mesmo turno — duas entradas heterogêneas por
    interação, com rótulos e scores inconsistentes entre si. Este módulo
    agora apenas RETORNA o resultado (kl, surprise, score sugerido); quem
    decide se e como consolidar é o chamador, em uma única escrita por
    turno.
    """

    def __init__(self, dim_state: int = 3, latent_dim: int = 2,
                 threshold_kl: float = 0.5, device="cpu"):
        super().__init__()
        self.threshold_kl = threshold_kl

        self.encoder = nn.Sequential(
            nn.Linear(dim_state, 8), nn.ReLU(), nn.Linear(8, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, dim_state)
        )

        # A distribuição de referência vive no mesmo espaço latente de q(z|x).
        # Usar dim_state aqui quebrava quando latent_dim != dim_state (o padrão
        # da classe principal é dim_state=3 e latent_dim=2).
        self.mem_mean = nn.Parameter(torch.zeros(latent_dim))
        self.mem_logvar = nn.Parameter(torch.zeros(latent_dim))

        self.device = device
        self.to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        var_q, var_p = torch.exp(logvar_q), torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
        return kl.sum()

    def forward(self, state: torch.Tensor, neuro_state: Dict) -> Dict:
        h = self.encoder(state.unsqueeze(0))
        mu_q, logvar_q = h[:, :2], h[:, 2:]

        mu_p, logvar_p = self.mem_mean, self.mem_logvar
        kl = self.kl_divergence(mu_q.squeeze(), logvar_q.squeeze(), mu_p, logvar_p)

        elbo = None
        surprise = kl.item() > self.threshold_kl

        if surprise:
            z = self.reparameterize(mu_q, logvar_q)
            recon = self.decoder(z)
            recon_loss = F.mse_loss(recon, state.unsqueeze(0), reduction="sum")
            elbo = (recon_loss + kl).item()

            with torch.no_grad():
                lr_mem = 0.01
                self.mem_mean.data += lr_mem * (mu_q.squeeze() - self.mem_mean.data)
                self.mem_logvar.data += lr_mem * (logvar_q.squeeze() - self.mem_logvar.data)

        L_val, Q_val, F_val = state[0].item(), state[1].item(), state[2].item()
        E_val = 0.5 * L_val + 0.5 * Q_val

        return {
            "kl": kl.item(),
            "elbo": elbo,
            "surprise": surprise,
            "E": E_val, "L": L_val, "Q": Q_val, "F": F_val,
            "score_sugerido": 0.8 if surprise else 0.2,
            "label_sugerido": "surpresa" if surprise else "estavel",
            "mu_q": mu_q.detach().squeeze().tolist(),
            "mu_p": mu_p.detach().tolist(),
        }

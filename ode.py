import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

try:
    from .neuroquimica import NeuroTransmissor
except (ImportError, ValueError):
    from neuroquimica import NeuroTransmissor


class QualiaODE(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.5, beta_L=0.3, beta_Q=0.3, lambda_=0.2,
                 kappa_L=0.1, kappa_Q=0.1, rho=0.15, mu=0.08, k_H=0.4,
                 sigma_0=0.3, eta_sigma=0.5, E0=5.0, eta_E=2.0, k_c=0.6,
                 coupling_strength_neuro: float = 0.05):
        super().__init__()

        self.params = nn.ParameterDict({
            "alpha": nn.Parameter(torch.tensor(alpha)),
            "gamma": nn.Parameter(torch.tensor(gamma)),
            "beta_L": nn.Parameter(torch.tensor(beta_L)),
            "beta_Q": nn.Parameter(torch.tensor(beta_Q)),
            "lambda_": nn.Parameter(torch.tensor(lambda_)),
            "kappa_L": nn.Parameter(torch.tensor(kappa_L)),
            "kappa_Q": nn.Parameter(torch.tensor(kappa_Q)),
            "rho": nn.Parameter(torch.tensor(rho)),
            "mu": nn.Parameter(torch.tensor(mu)),
            "k_H": nn.Parameter(torch.tensor(k_H)),
        })

        self.neuro = NeuroTransmissor()
        self.sigma_0 = sigma_0
        self.eta_sigma = eta_sigma
        self.E0 = E0
        self.eta_E = eta_E
        self.k_c = k_c
        self.w_L = 0.5
        self.w_Q = 0.5
        # Peso pequeno por padrão: os 5 canais antes não usados passam a
        # influenciar (L, Q) de fato, mas sem desestabilizar a dinâmica já
        # calibrada pelos 3 canais originais (dopamina/serotonina/noradrenalina).
        self.coupling_strength_neuro = coupling_strength_neuro

        # Canais que antes tinham matriz de modulação calculada e nunca
        # consumida pela ODE (CORREÇÃO: agora entram via produto matriz-vetor
        # real contra [L, Q], não apenas como elementos diagonais isolados).
        self._canais_extra = ["adrenalina", "gaba", "ocitocina", "acetilcolina", "glutamato"]

    def E_crit(self, a: float) -> float:
        return self.E0 + self.eta_E * a

    def sigma(self, a: float) -> float:
        return self.sigma_0 * np.exp(-self.eta_sigma * a)

    def forward(self, t, X, H_t=0.0, eta_t=0.0, a=0.5):
        L, Q, F = X[0], X[1], X[2]
        E = self.w_L * L + self.w_Q * Q
        LQ = torch.stack([L, Q])

        mods = self.neuro.get_modulation_matrices()
        mod_d, mod_s, mod_n = mods["dopamina"], mods["serotonina"], mods["noradrenalina"]

        alpha_mod = self.params["alpha"] * (1.0 + mod_n[0, 0])
        gamma_mod = self.params["gamma"] * (1.0 + mod_d[0, 0])
        beta_L_mod = self.params["beta_L"] * (1.0 + mod_s[0, 0])

        # Multiplicação matriz-vetor real sobre (L, Q) para os 5 canais extra
        extra_LQ = torch.zeros(2, dtype=LQ.dtype)
        for nome in self._canais_extra:
            extra_LQ = extra_LQ + mods[nome] @ LQ
        extra_LQ = extra_LQ * self.coupling_strength_neuro

        dL_dt = (torch.tanh(alpha_mod * Q) + gamma_mod * Q - beta_L_mod * L -
                 self.params["lambda_"] * E - self.params["kappa_L"] * F + extra_LQ[0])

        dQ_dt = (torch.tanh(alpha_mod * L) + self.params["gamma"] * L -
                 self.params["beta_Q"] * Q - self.params["lambda_"] * E -
                 self.params["kappa_Q"] * F + self.params["k_H"] * H_t +
                 self.sigma(a) * eta_t + extra_LQ[1])

        dF_dt = self.params["rho"] * E - self.params["mu"] * F

        return torch.stack([dL_dt, dQ_dt, dF_dt])

    def evolve(self, X0, t_span, H_func: Optional[Callable] = None, a: float = 0.5, eta_base=0.0):
        """
        CORREÇÃO: antes, `random.gauss` era resorteado a cada chamada interna
        de `ode_func`, o que o RK4 invoca várias vezes por passo (estágios
        intermediários) — produzindo ruído inconsistente com o esquema de
        integração, sem semântica de processo de Wiener.

        Aqui sorteamos UM valor de ruído por intervalo entre pontos de
        `t_span` (ou seja, por passo de saída), e o mantemos fixo durante
        os estágios internos do RK4 daquele intervalo. Isso não é um
        solver de SDE rigoroso (para isso, usar `torchsde`), mas é
        consistente e reprodutível dado um mesmo `t_span`.
        """
        eta_base_val = eta_base.item() if torch.is_tensor(eta_base) else float(eta_base)
        n_steps = max(len(t_span) - 1, 1)
        ruidos = [random.gauss(eta_base_val, 1.0) for _ in range(n_steps)]
        t0, t_total = float(t_span[0]), float(t_span[-1] - t_span[0]) + 1e-8

        def ode_func(t, X):
            H = H_func(t) if H_func is not None else 0.0
            idx = int((float(t) - t0) / t_total * n_steps)
            idx = max(0, min(idx, n_steps - 1))
            eta = ruidos[idx]
            return self.forward(t, X, H, eta, a)

        return odeint(ode_func, X0, t_span, method="rk4")

    def apply_collapse(self, X, E, E_crit):
        if E > E_crit:
            L, Q, F = X[0], X[1], X[2]
            return torch.stack([
                L * torch.exp(-torch.tensor(self.k_c)),
                Q * torch.exp(-torch.tensor(self.k_c)),
                F + 2.0,
            ]), True
        return X, False

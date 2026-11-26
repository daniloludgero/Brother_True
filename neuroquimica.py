import random
from typing import Dict, Optional

import torch

try:
    from .farmacologia import CONSTANTES_NEUROQUIMICAS, hill_occupancy
except (ImportError, ValueError):
    from farmacologia import CONSTANTES_NEUROQUIMICAS, hill_occupancy

# Ordem fixa usada pela matriz de acoplamento (índices consistentes em todo o módulo)
ORDEM_NEUROTRANSMISSORES = [
    "dopamina", "serotonina", "noradrenalina", "adrenalina",
    "gaba", "ocitocina", "acetilcolina", "glutamato",
]


def _construir_matriz_acoplamento() -> torch.Tensor:
    """
    Matriz 8x8 de acoplamento entre neurotransmissores.

    As cinco primeiras entradas reproduzem EXATAMENTE as relações que já
    existiam, escritas à mão, na versão anterior de `atualizar()`:
        gaba    -= 0.15 * glutamato
        gaba    += 0.10 * acetilcolina
        acetilcolina += 0.12 * dopamina
        glutamato    += 0.18 * noradrenalina
        glutamato    -= 0.25 * gaba

    Isto é apenas uma reorganização em forma matricial — os números não
    mudaram — para que a multiplicação seja de fato matriz-vetor, e não
    extração de elementos isolados. As duas últimas entradas são extensões
    ilustrativas (claramente novas, pensadas para refinamento futuro):
    ocitocina amortecendo o eixo de estresse, e adrenalina aguda reduzindo
    serotonina. Ajuste ou remova livremente.
    """
    n = len(ORDEM_NEUROTRANSMISSORES)
    idx = {nome: i for i, nome in enumerate(ORDEM_NEUROTRANSMISSORES)}
    M = torch.zeros(n, n)

    M[idx["gaba"], idx["glutamato"]] = -0.15
    M[idx["gaba"], idx["acetilcolina"]] = 0.10
    M[idx["acetilcolina"], idx["dopamina"]] = 0.12
    M[idx["glutamato"], idx["noradrenalina"]] = 0.18
    M[idx["glutamato"], idx["gaba"]] = -0.25

    # Extensões ilustrativas (novas, não presentes no código original):
    M[idx["ocitocina"], idx["adrenalina"]] = -0.08
    M[idx["adrenalina"], idx["serotonina"]] = -0.10

    return M


class NeuroTransmissor:
    def __init__(self, constantes: Optional[Dict] = None, coupling_dt: float = 1.0):
        constantes = constantes or CONSTANTES_NEUROQUIMICAS

        def repouso(nome: str) -> torch.Tensor:
            c = constantes[nome]
            c_inicial = c.c_inicial_nm if c.c_inicial_nm is not None else c.kd_nm
            return torch.tensor(hill_occupancy(c_inicial, c.kd_nm, c.n_hill), dtype=torch.float32)

        # Baselines agora derivados da equação de Hill sobre concentrações
        # reais (CORREÇÃO: antes eram constantes escolhidas à mão).
        self.dopamina = repouso("dopamina")
        self.serotonina = repouso("serotonina")
        self.noradrenalina = repouso("noradrenalina")
        self.adrenalina = repouso("adrenalina")
        self.gaba = repouso("gaba")
        self.ocitocina = repouso("ocitocina")
        self.acetilcolina = repouso("acetilcolina")
        self.glutamato = repouso("glutamato")

        self.coupling_matrix = _construir_matriz_acoplamento()
        self.coupling_dt = coupling_dt
        self.historico = []

    def _vetor_estado(self) -> torch.Tensor:
        return torch.stack([
            self.dopamina, self.serotonina, self.noradrenalina, self.adrenalina,
            self.gaba, self.ocitocina, self.acetilcolina, self.glutamato,
        ])

    def atualizar(self, E: float, L: float, Q: float, collapsed: bool = False):
        intensidade = E / 12.0
        valencia = (L + Q) / 2.0

        # --- Heurísticas primárias dirigidas por E/L/Q (mantidas sem alteração) ---
        delta_d = intensidade * (1.2 if valencia > 0.6 else -0.4)
        self.dopamina = torch.clamp(self.dopamina + delta_d, 0.0, 1.0)
        self.serotonina = torch.clamp(self.serotonina + 0.08 * (1.0 - abs(valencia - 0.5)), 0.2, 0.95)
        self.noradrenalina = torch.clamp(0.35 + 0.25 * torch.randn(1).clamp(-1, 1), 0.1, 0.8).squeeze()
        delta_ad = intensidade * (-0.8 if (valencia < 0.4 or collapsed) else 0.3)
        self.adrenalina = torch.clamp(self.adrenalina + delta_ad, 0.0, 0.9)
        self.gaba = torch.clamp(self.gaba + 0.12 * (1.0 - self.adrenalina), 0.4, 0.95)
        delta_o = intensidade * (1.1 if valencia > 0.7 else -0.3)
        self.ocitocina = torch.clamp(self.ocitocina + delta_o, 0.1, 1.0)
        delta_ach = intensidade * (0.9 if valencia > 0.5 else -0.25)
        self.acetilcolina = torch.clamp(self.acetilcolina + delta_ach, 0.2, 0.95)
        delta_glu = intensidade * (1.15 if L > Q * 0.8 else 0.4) - 0.2 * self.gaba
        self.glutamato = torch.clamp(self.glutamato + delta_glu, 0.3, 1.0)

        # --- Acoplamento cruzado real, via multiplicação matriz-vetor ---
        # (CORREÇÃO: antes eram três linhas manuais; agora é uma única
        # operação que pode ser estendida adicionando entradas à matriz.)
        estado_vec = self._vetor_estado()
        delta_acoplamento = self.coupling_matrix @ estado_vec
        novo_estado = torch.clamp(estado_vec + delta_acoplamento * self.coupling_dt, 0.0, 1.0)

        (self.dopamina, self.serotonina, self.noradrenalina, self.adrenalina,
         self.gaba, self.ocitocina, self.acetilcolina, self.glutamato) = [
            novo_estado[i] for i in range(novo_estado.shape[0])
        ]

        self.historico.append({**self.estado_atual(), "E": E, "L": L, "Q": Q})

    def estado_atual(self) -> Dict[str, float]:
        return {
            "dopamina": self.dopamina.item(), "serotonina": self.serotonina.item(),
            "noradrenalina": self.noradrenalina.item(), "adrenalina": self.adrenalina.item(),
            "gaba": self.gaba.item(), "ocitocina": self.ocitocina.item(),
            "acetilcolina": self.acetilcolina.item(), "glutamato": self.glutamato.item(),
        }

    def get_modulation_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Matrizes 2x2 usadas para modular (L, Q) dentro da ODE.

        Dopamina/serotonina/noradrenalina seguem o uso original (extração
        do elemento diagonal para modular alpha/gamma/beta_L). Os outros
        cinco canais — antes definidos aqui mas nunca consumidos pela ODE —
        agora são de fato multiplicados pelo vetor [L, Q] dentro de
        QualiaODE.forward (ver ode.py), em vez de ficarem como código morto.
        """
        return {
            "dopamina": torch.tensor([[self.dopamina, 0.0], [0.0, self.dopamina * 0.8]]),
            "serotonina": torch.tensor([[self.serotonina * 0.9, 0.0], [0.0, self.serotonina]]),
            "noradrenalina": torch.tensor([[self.noradrenalina * 0.7, 0.0], [0.0, self.noradrenalina * 1.15]]),
            "adrenalina": torch.tensor([[self.adrenalina * -0.6, 0.0], [0.0, self.adrenalina * 1.3]]),
            "gaba": torch.tensor([[self.gaba * 0.85, 0.0], [0.0, self.gaba * 1.1]]),
            "ocitocina": torch.tensor([[self.ocitocina * 1.2, 0.1 * self.ocitocina], [0.1 * self.ocitocina, self.ocitocina * 0.9]]),
            "acetilcolina": torch.tensor([[self.acetilcolina * 1.25, 0.0], [0.0, self.acetilcolina * 1.1]]),
            "glutamato": torch.tensor([[self.glutamato * 1.35, 0.0], [0.0, self.glutamato * 0.75]]),
        }

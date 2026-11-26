# ================================================================
# FARMACOLOGIA: Constantes reais e equação de ocupação de receptor
# ================================================================
#
# Substitui os valores ad hoc (ex.: dopamina = 0.55) por uma derivação
# baseada em faixas extracelulares reportadas na literatura. Em vez de
# inserir as concentrações brutas (que variam ~6 ordens de grandeza
# entre si, o que quebraria a escala [0,1] usada em todo o resto do
# sistema), convertemos cada concentração em uma FRAÇÃO DE OCUPAÇÃO DE
# RECEPTOR via equação de Hill:
#
#       ocupação(C) = C^n / (Kd^n + C^n)
#
# onde Kd é a concentração de meia-saturação (aproximada aqui pelo
# "valor central" da faixa fisiológica) e n é o coeficiente de Hill
# (1.0 = sem cooperatividade; pode ser refinado por substância depois).
#
# Por construção, quando C == Kd a ocupação é exatamente 0.5 — ou seja,
# o "estado de repouso" de cada neurotransmissor passa a ser 0.5 por
# definição, não por escolha arbitrária. Isso troca a calibração ad hoc
# por uma calibração rastreável a uma referência fisiológica, sem exigir
# reescrever as equações de acoplamento já existentes (que continuam
# operando sobre números em [0,1]).
#
# Todas as concentrações abaixo estão em nM (nanomolar), já convertidas
# pelo autor a partir das faixas originais (μM, nM, pM).

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ConstantesNeuroquimicas:
    nome: str
    c_min_nm: float       # menor concentração extracelular tipicamente reportada
    kd_nm: float          # valor central da faixa == aproximação do Kd (meia-saturação)
    c_max_nm: float        # maior concentração extracelular tipicamente reportada
    n_hill: float = 1.0
    # Concentração inicial usada para calcular o estado de repouso.
    # Se None, usa kd_nm (-> ocupação inicial = 0.5 exatamente).
    # Pode ser ajustada para reproduzir baselines comportamentais antigos,
    # ex.: GABA com tônus de repouso um pouco acima do Kd.
    c_inicial_nm: Optional[float] = None


# Tabela construída diretamente a partir dos valores fornecidos (já em nM)
CONSTANTES_NEUROQUIMICAS: Dict[str, ConstantesNeuroquimicas] = {
    "glutamato":     ConstantesNeuroquimicas("glutamato", 30, 2000, 5000),
    "gaba":          ConstantesNeuroquimicas("gaba", 200, 500, 800),
    "acetilcolina":  ConstantesNeuroquimicas("acetilcolina", 10, 500, 3000),
    "dopamina":      ConstantesNeuroquimicas("dopamina", 14, 100, 700),
    "noradrenalina": ConstantesNeuroquimicas("noradrenalina", 14, 40, 78),
    "serotonina":    ConstantesNeuroquimicas("serotonina", 1, 20, 100),
    "adrenalina":    ConstantesNeuroquimicas("adrenalina", 0.01, 0.1, 1),
    "ocitocina":     ConstantesNeuroquimicas("ocitocina", 0.001, 0.01, 0.1),
}


def hill_occupancy(c: float, kd: float, n: float = 1.0) -> float:
    """Fração de ocupação de receptor (0 a 1) via equação de Hill."""
    if c <= 0:
        return 0.0
    cn = c ** n
    return cn / (kd ** n + cn)


def estado_repouso(nome: str, constantes: Optional[Dict[str, ConstantesNeuroquimicas]] = None) -> float:
    """Calcula a ocupação de repouso (~0.5 por padrão) para um neurotransmissor."""
    constantes = constantes or CONSTANTES_NEUROQUIMICAS
    c = constantes[nome]
    c_inicial = c.c_inicial_nm if c.c_inicial_nm is not None else c.kd_nm
    return hill_occupancy(c_inicial, c.kd_nm, c.n_hill)


# Exemplo de como recuperar um baseline assimétrico antigo, se desejado,
# em vez do padrão 0.5 universal (descomente e ajuste conforme necessário):
#
# CONSTANTES_NEUROQUIMICAS["gaba"].c_inicial_nm = 650.0   # -> ocupação ~0.57
# CONSTANTES_NEUROQUIMICAS["adrenalina"].c_inicial_nm = 0.04  # -> ocupação ~0.29

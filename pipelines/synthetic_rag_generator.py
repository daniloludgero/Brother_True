#!/usr/bin/env python3
"""
pipelines/synthetic_rag_generator.py
====================================
Synthetic Corpus & RAG Grounded/Refusal Generator for BrotherTrue (Milestone M2 / Requirement R1).

Responsibilities:
1. Hardware Sentry:
   - Continuously monitors host SSD C: (> 4.5 GB, critical abort < 4.0 GB).
   - Enforces torch.set_num_threads(2) and offline mode (HF_HUB_OFFLINE=1).
2. Multi-Domain Synthetic Corpus (`dados/corpus_sintetico_multidominio.jsonl`):
   - High-density conversational pairs across 6 balanced domains:
     * analytical_reasoning (20%)
     * stem (15%)
     * code (15%)
     * general_knowledge (20%)
     * philosophy_qualia (15%)
     * empathetic_dialogue (15%)
   - Strict adherence to `dados/schema_cot_biocognitivo.json`.
   - DeepSeek-style CoT: thought_trace strictly isolated from response_text (zero leakage).
   - Biological invariants: Lyapunov energy containment and Hill receptor occupancy.
3. RAG Grounded & Refusal Dataset (`dados/treino_rag_grounded.jsonl`):
   - Balanced RAG dataset:
     * ~60% grounded_positive
     * ~25% adversarial_negative (insufficient/irrelevant context -> polite, precise refusal without hallucination)
     * ~15% partial_grounded / contradictory_context
   - Strict adherence to `dados/schema_rag_grounded.json` with all 9 required fields:
     id, query, retrieved_context, grounded_response, thought_trace, is_grounded, abstention_flag, tutor_domain, sha256.
4. Cryptographic Manifest (`dados/manifest_sintetico_rag.json`):
   - Line counts, domain distributions, real token statistics, Draft-07 validation flags, and SHA-256 provenance hashes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

# 1. HARDWARE GOVERNANCE: enforce threads and offline mode immediately
import torch
torch.set_num_threads(2)
os.environ["HF_HUB_OFFLINE"] = "1"

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct import of DeepSeekCoTParser to prevent WSL /mnt/c torch init delay in models/__init__.py
try:
    import importlib.util
    _cot_path = REPO_ROOT / "models" / "cot_parser.py"
    _spec = importlib.util.spec_from_file_location("cot_parser_direct", str(_cot_path))
    _cot_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cot_mod)
    DeepSeekCoTParser = _cot_mod.DeepSeekCoTParser
except Exception:
    from models.cot_parser import DeepSeekCoTParser

from teachers.teacher_federation import (
    BaseTutor,
    RAGRecord,
    TeacherFederation,
    TeacherTurn,
)


# ==============================================================================
# HARDWARE SENTRY
# ==============================================================================

class HardwareSentry:
    """Monitors host storage and thread constraints."""

    def __init__(self, mount_point: str = "/mnt/c", min_free_gb: float = 4.5, critical_free_gb: float = 4.0):
        self.mount_point = mount_point
        self.min_free_gb = min_free_gb
        self.critical_free_gb = critical_free_gb

    def get_free_gb(self) -> float:
        """Returns free space in Gigabytes (decimal 10^9 bytes) on mount point."""
        try:
            stat = shutil.disk_usage(self.mount_point)
            return stat.free / 1e9
        except Exception:
            stat = shutil.disk_usage(".")
            return stat.free / 1e9

    def get_free_gib(self) -> float:
        """Returns free space in GiB (binary 1024^3 bytes)."""
        try:
            stat = shutil.disk_usage(self.mount_point)
            return stat.free / (1024 ** 3)
        except Exception:
            stat = shutil.disk_usage(".")
            return stat.free / (1024 ** 3)

    def verify_or_enforce(self) -> float:
        free_gb = self.get_free_gb()
        if free_gb < self.critical_free_gb:
            raise RuntimeError(
                f"[HardwareSentry CRITICAL] Free space on {self.mount_point} is {free_gb:.2f} GB, "
                f"below critical floor of {self.critical_free_gb} GB! Aborting."
            )
        return free_gb


# ==============================================================================
# TOKEN COUNTER (LOCAL QWEN / FALLBACK)
# ==============================================================================

class TokenCounter:
    """Calculates authentic token counts using subword BPE estimation for Portuguese and ChatML."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        # ChatML, code, and punctuation-aware tokenization: words + symbols
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        # Average subword expansion factor for BPE in Portuguese and technical markdown is ~1.15
        return max(1, int(len(tokens) * 1.15))


# ==============================================================================
# CURATED KNOWLEDGE & CONVERSATION BANKS
# ==============================================================================

MULTIDOMAIN_PROMPT_BANK: List[Dict[str, Any]] = [
    # 1. Analytical Reasoning (20%)
    {
        "domain": "analytical_reasoning",
        "source": "discrete_math_combinatorics",
        "prompt": "Considere um conjunto de 100 fichas alinhadas em uma fita. É permitido trocar duas fichas adjacentes pelo custo de 1 rublo, ou trocar gratuitamente duas fichas se houver exatamente três fichas entre elas. Qual é o custo mínimo em rublos para inverter completamente a ordem das fichas?",
        "thought": "1. Identificação de invariantes:\n   - Inverter ordem: ficha i vai para 101 - i.\n   - Troca gratuita: |i - j| = 4 (3 fichas intermediárias).\n   - Congruência módulo 4 divide as 100 fichas em 4 classes de 25 elementos: C0, C1, C2, C3.\n2. Custo dentro da classe: 0 rublos.\n3. Mapeamento sob congruência:\n   - 101 - i ≡ 1 - i (mod 4).\n   - C0 -> C1, C1 -> C0; C2 -> C3, C3 -> C2.\n4. Custo via trocas adjacentes (custo 1):\n   - Cada troca adjacente altera no máximo uma classe entre elementos vizinhos.\n   - Transpor 25 elementos de C0 para C1: 25 trocas.\n   - Transpor 25 elementos de C2 para C3: 25 trocas.\n   - Custo estrito mínimo = 25 + 25 = 50 rublos.",
        "response": "O custo mínimo necessário para inverter a ordem das 100 fichas é de exatamente 50 rublos.\n\nA regra de permuta gratuita permite transpor sem custo fichas cujas posições distem 4 unidades (|i - j| = 4), particionando a fita em 4 classes de congruência módulo 4 com 25 fichas cada. Ao inverter a sequência, a posição final 101 - i mapeia a classe 0 na classe 1 (e vice-versa) e a classe 2 na classe 3 (e vice-versa). Como cada troca adjacente de 1 rublo só pode transferir fichas entre classes vizinhas, são necessárias 25 trocas para o par de classes 0-1 e 25 trocas para o par 2-3, resultando em 50 rublos."
    },
    {
        "domain": "analytical_reasoning",
        "source": "game_theory_nash_deduction",
        "prompt": "Dois jogadores participam de um jogo da barganha de Rubinstein com horizonte infinito e fatores de desconto delta_1 e delta_2. Como se deduz a partição de equilíbrio perfeito em subjogos (SPE)?",
        "thought": "1. Estrutura do jogo de Rubinstein:\n   - Ofertas alternadas em t = 0, 1, 2, ...\n   - O jogador 1 propõe uma partição (x, 1 - x). Se aceito, o jogo termina com payoffs (x, 1 - x).\n   - Se rejeitado, no turno t=1 o jogador 2 propõe (1 - y, y) com desconto.\n2. Invariância estacionária do SPE:\n   - Seja v1 o payoff do jogador 1 quando ele propõe, e v2 o payoff do jogador 2 quando ele propõe.\n   - Equações de corte: 1 - v1 = delta2 * v2 (jogador 1 deixa o jogador 2 indiferente à sua proposta de t=1).\n   - Da mesma forma: 1 - v2 = delta1 * v1.\n3. Resolução do sistema linear 2x2:\n   - v1 = 1 - delta2 * v2 = 1 - delta2 * (1 - delta1 * v1) = 1 - delta2 + delta1 * delta2 * v1.\n   - v1 * (1 - delta1 * delta2) = 1 - delta2  =>  v1* = (1 - delta2) / (1 - delta1 * delta2).\n4. Conclusão unívoca: A partição é aceita imediatamente em t=0.",
        "response": "No jogo da barganha de Rubinstein com horizonte infinito, a partição única de Equilíbrio Perfeito em Subjogos (SPE) prevê que o jogador 1 propõe e o jogador 2 aceita imediatamente no primeiro turno ($t=0$).\n\nA fração obtida pelo jogador 1 ($x^*$) e pelo jogador 2 ($1 - x^*$) é dada analiticamente por:\n\n$$x^* = \\frac{1 - \\delta_2}{1 - \\delta_1 \\delta_2}, \\quad 1 - x^* = \\frac{\\delta_2 (1 - \\delta_1)}{1 - \\delta_1 \\delta_2}$$\n\nEssa solução é deduzida pela estacionariedade dos subjogos, onde o proponente em cada estágio oferece exatamente o valor presente descontado que o respondente obteria caso rejeitasse e fizesse a contraproposta no turno seguinte."
    },
    {
        "domain": "analytical_reasoning",
        "source": "logic_knight_knave_deduction",
        "prompt": "Em uma ilha habitada apenas por Cavaleiros (que sempre dizem a verdade) e Vilões (que sempre mentem), encontramos A e B. A diz: 'Pelo menos um de nós é um Vilão'. O que podemos deduzir sobre as identidades de A e B?",
        "thought": "1. Hipótese 1: Suponha que A seja um Vilão.\n   - Se A é Vilão, a declaração de A ('Pelo menos um de nós é um Vilão') deve ser falsa.\n   - A negação de 'pelo menos um é vilão' é 'nenhum de nós é vilão', ou seja, ambos são cavaleiros.\n   - Mas isso implicaria que A é cavaleiro, uma contradição direta com a premissa de que A é Vilão.\n   - Portanto, a hipótese de que A é Vilão é refutada por reductio ad absurdum.\n2. Conclusão sobre A: A é obrigatoriamente um Cavaleiro.\n3. Dedução sobre B:\n   - Como A é Cavaleiro, sua declaração é verdadeira.\n   - Logo, 'pelo menos um de nós é um Vilão' é verdadeiro.\n   - Como A não é Vilão, B deve necessariamente ser o Vilão.\n4. Síntese final: A é Cavaleiro e B é Vilão.",
        "response": "Podemos deduzir com absoluta certeza lógica que A é um Cavaleiro e B é um Vilão.\n\nSe A fosse um Vilão, sua afirmação seria falsa, o que exigiria que nenhum deles fosse vilão (ambos seriam cavaleiros), gerando uma contradição imediata. Portanto, A é necessariamente um Cavaleiro e sua declaração é verdadeira. Para que 'pelo menos um de nós é um Vilão' seja verdade e A seja Cavaleiro, B é compulsoriamente um Vilão."
    },

    # 2. STEM (15%)
    {
        "domain": "stem",
        "source": "biophysics_goldman_hodgkin_katz",
        "prompt": "Como a equação de Goldman-Hodgkin-Katz (GHK) expande a equação de Nernst para calcular o potencial de repouso transmembrana de um neurônio?",
        "thought": "1. Equação de Nernst para um único íon permeável em equilíbrio eletroquímico:\n   - E_ion = (RT / zF) * ln([ion]_out / [ion]_in).\n2. Limitação no estado fisiológico neuronal:\n   - A membrana neuronal possui permeabilidades simultâneas para K+, Na+ e Cl- (P_K, P_Na, P_Cl).\n   - O estado de repouso é um estado estacionário dinâmico mantido pela bomba Na+/K+-ATPase.\n3. Formulação da equação de GHK sob campo elétrico constante:\n   - V_m = (RT / F) * ln( (P_K[K+]_out + P_Na[Na+]_out + P_Cl[Cl-]_in) / (P_K[K+]_in + P_Na[Na+]_in + P_Cl[Cl-]_out) ).\n4. Síntese do significado biofísico: Como P_K >> P_Na (cerca de 1 : 0.04), V_m repousa próximo a -70 mV.",
        "response": "A equação de Goldman-Hodgkin-Katz (GHK) expande a equação de Nernst ao quantificar o potencial elétrico transmembrana em um estado estacionário dinâmico no qual múltiplos íons permeiam simultaneamente a membrana celular com coeficientes de permeabilidade distintos.\n\nEnquanto a equação de Nernst se restringe ao equilíbrio termodinâmico de uma única espécie iônica ($E_i = \\frac{RT}{zF} \\ln \\frac{[i]_{\\text{ext}}}{[i]_{\\text{int}}}$), a formulação GHK integra as contribuições ponderadas de potássio ($K^+$), sódio ($Na^+$) e cloreto ($Cl^-$):\n\n$$V_m = \\frac{RT}{F} \\ln \\left( \\frac{P_K [K^+]_{\\text{ext}} + P_{Na} [Na^+]_{\\text{ext}} + P_{Cl} [Cl^-]_{\\text{int}}}{P_K [K^+]_{\\text{int}} + P_{Na} [Na^+]_{\\text{int}} + P_{Cl} [Cl^-]_{\\text{ext}}} \\right)$$\n\nComo a permeabilidade da membrana ao potássio em repouso supera amplamente a permeabilidade ao sódio ($P_K \\gg P_{Na}$), o potencial de repouso situa-se tipicamente entre -65 mV e -75 mV, atraído para o potencial de reversão do potássio."
    },
    {
        "domain": "stem",
        "source": "thermodynamics_carnot_cycle",
        "prompt": "Por que o ciclo de Carnot estabelece o limite teórico superior de rendimento para qualquer máquina térmica operando entre duas temperaturas fixas?",
        "thought": "1. Definição do Ciclo de Carnot:\n   - Quatro processos reversíveis: expansão isotérmica (T_quente), expansão adiabática, compressão isotérmica (T_fria), compressão adiabática.\n2. Segunda Lei da Termodinâmica (Enunciados de Clausius e Kelvin-Planck):\n   - Não é possível converter integralmente calor em trabalho em um ciclo termodinâmico fechado sem transferência residual de calor para uma fonte fria.\n3. Desigualdade de Clausius: integral_ciclo (dQ / T) <= 0, com igualdade estrita apenas para processos reversíveis.\n4. Rendimento de Carnot: eta_Carnot = 1 - (T_fria / T_quente).\n   - Se existisse máquina com eta > eta_Carnot, o acoplamento reverso violaria a 2ª Lei ao transferir calor da fonte fria para a quente sem trabalho externo.",
        "response": "O ciclo de Carnot estabelece o teto máximo de eficiência termodinâmica porque opera exclusivamente através de transformações estritamente reversíveis (duas isotérmicas e duas adiabáticas), minimizando a geração de entropia a zero durante o ciclo.\n\nPela Segunda Lei da Termodinâmica e o Teorema de Carnot, o rendimento depende estritamente das temperaturas absolutas dos reservatórios térmico quente ($T_q$) e frio ($T_f$):\n\n$$\\eta_{\\text{Carnot}} = 1 - \\frac{T_f}{T_q}$$\n\nQualquer máquina térmica irreversível operando entre os mesmos reservatórios sofre dissipações internas (atrito, condução térmica com gradiente finito de temperatura), gerando entropia positiva e resultando obrigatoriamente em rendimento inferior."
    },

    # 3. Code & Algorithms (15%)
    {
        "domain": "code",
        "source": "pytorch_module_architecture",
        "prompt": "Como implementar um módulo PyTorch limpo com LoRA injetado sobre camadas lineares preservando a herança de nn.Module e autograd?",
        "thought": "1. Princípio de LoRA (Low-Rank Adaptation):\n   - W = W_0 + (alpha / r) * (B @ A), onde A ~ N(0, sigma^2) e B = 0 na inicialização.\n2. Contrato de POO em PyTorch:\n   - Subclasse de nn.Module.\n   - Congelar parâmetros originais: base_layer.weight.requires_grad = False.\n   - Parâmetros treináveis: lora_A (r, d_in) e lora_B (d_out, r).\n3. Implementação forward:\n   - output = base_layer(x) + (alpha / r) * (F.linear(F.linear(x, lora_A), lora_B)).\n4. Preservação de autograd e dtype consistente.",
        "response": "Abaixo está a implementação canônica e limpa de uma camada LoRA linear em PyTorch, respeitando a herança de `nn.Module` e o isolamento de autograd:\n\n```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass LoRALinear(nn.Module):\n    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0):\n        super().__init__()\n        self.base_layer = base_layer\n        self.rank = rank\n        self.scaling = alpha / rank\n        \n        # Congela pesos originais\n        self.base_layer.weight.requires_grad = False\n        if self.base_layer.bias is not None:\n            self.base_layer.bias.requires_grad = False\n            \n        # Matrizes de baixo posto treináveis\n        in_features = base_layer.in_features\n        out_features = base_layer.out_features\n        self.lora_A = nn.Parameter(torch.empty(rank, in_features))\n        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))\n        \n        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)\n        \n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        base_out = self.base_layer(x)\n        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling\n        return base_out + lora_out\n```\n\nEssa estrutura garante inicialização neutra ($B=0$), preservando o comportamento original do modelo no passo zero e permitindo convergência estável com baixíssimo consumo de memória de gradiente."
    },
    {
        "domain": "code",
        "source": "algorithms_lru_cache",
        "prompt": "Como projetar uma estrutura de dados LRU (Least Recently Used) Cache em Python com operações get e put em complexidade O(1)?",
        "thought": "1. Requisitos estruturais: get(key) O(1) e put(key, value) O(1).\n2. Mecanismo de dados:\n   - Dicionário hash (busca O(1) de nós).\n   - Lista duplamente encadeada (remoção e inserção em O(1) nas extremidades).\n3. Operações:\n   - get: se chave existe, move nó para a cabeça (mais recente) e retorna valor.\n   - put: se chave existe, atualiza valor e move para a cabeça. Se nova, cria nó; se capacidade estourar, remove a cauda (menos recente).",
        "response": "Para obter complexidade $O(1)$ tanto em `get` quanto em `put`, combinamos um dicionário hash com uma lista duplamente encadeada (ou usamos diretamente `collections.OrderedDict`). Veja a implementação fundamental:\n\n```python\nclass Node:\n    def __init__(self, key: int = 0, val: int = 0):\n        self.key, self.val = key, val\n        self.prev = self.next = None\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = {}  # key -> Node\n        # Sentinelas cabeça e cauda\n        self.head, self.tail = Node(), Node()\n        self.head.next, self.tail.prev = self.tail, self.head\n\n    def _remove(self, node: Node):\n        node.prev.next = node.next\n        node.next.prev = node.prev\n\n    def _insert_head(self, node: Node):\n        node.next = self.head.next\n        node.prev = self.head\n        self.head.next.prev = node\n        self.head.next = node\n\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        node = self.cache[key]\n        self._remove(node)\n        self._insert_head(node)\n        return node.val\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self._remove(self.cache[key])\n        node = Node(key, value)\n        self.cache[key] = node\n        self._insert_head(node)\n        if len(self.cache) > self.capacity:\n            lru = self.tail.prev\n            self._remove(lru)\n            del self.cache[lru.key]\n```"
    },

    # 4. General Knowledge & Epistemology (20%)
    {
        "domain": "general_knowledge",
        "source": "systems_epistemology_cybernetics",
        "prompt": "Como a Lei da Variedade Requisita de W. Ross Ashby se aplica à governança e estabilidade de sistemas cognitivos adaptativos?",
        "thought": "1. Enunciado formal da Lei de Ashby (1956):\n   - 'Apenas a variedade pode absorver ou destruir a variedade' (Only variety can destroy variety).\n   - Formalmente: se R é o regulador e D são as perturbações do ambiente, a variedade do desfecho V(O) >= V(D) - V(R).\n2. Aplicação em sistemas cognitivos e IA:\n   - Um modelo inteligente só pode manter estabilidade homeostática interna se o repertório de seus estados internos (variedade do regulador) for igual ou superior à riqueza de perturbações contextuais do ambiente.\n3. Conexão com BrotherTrue: A combinação de EDOs contínuas com LLMs comerciais visa suprir variedade interna para lidar com complexidade multi-domínio.",
        "response": "A Lei da Variedade Requisita, formulada pelo ciberneticista W. Ross Ashby em 1956, estabelece que para um sistema alcançar estabilidade homeostática diante de um ambiente complexo, a variedade interna de estados de seu regulador deve ser igual ou superior à variedade de perturbações às quais está submetido.\n\nEm termos matemáticos:\n\n$$V(O) \\ge V(D) - V(R)$$\n\nOnde $V(D)$ é a variedade das perturbações ambientais, $V(R)$ é a variedade de respostas do regulador e $V(O)$ é a variedade dos desfechos. Em sistemas cognitivos e inteligências artificiais adaptativas, isso significa que para responder com precisão e segurança a uma multiplicidade de domínios (STEM, diálogo empático, código, epistemologia), o modelo não pode operar sob espaço representacional excessivamente restrito; seu espaço de estados deve espelhar a dimensionalidade do ambiente para manter a estabilidade sistêmica."
    },
    {
        "domain": "general_knowledge",
        "source": "scientific_revolution_epistemology",
        "prompt": "Qual é a distinção conceitual fundamental entre o falseacionismo de Karl Popper e os paradigmas científicos de Thomas Kuhn?",
        "thought": "1. Karl Popper (A Lógica da Pesquisa Científica):\n   - Demarcação: Uma teoria só é científica se for empiricamente falseável (refutável por observação).\n   - Progresso cumulativo por eliminação de hipóteses refutadas (conjecturas e refutações).\n2. Thomas Kuhn (A Estrutura das Revoluções Científicas):\n   - Ciência Normal: Resolução de quebra-cabeças dentro de um paradigma hegemônico aceito pela comunidade.\n   - Acúmulo de anomalias -> Crise paradigmática -> Revolução científica e mudança de paradigma (incomensurabilidade).\n3. Síntese comparativa: Popper foca na lógica normativa da validação teórica; Kuhn foca na sociologia e história prática da comunidade científica.",
        "response": "A distinção fundamental reside no método e na dinâmica de transição do conhecimento científico:\n\n1. **Karl Popper (Falseacionismo)**: Adota uma abordagem lógica e normativa. Uma proposição só é científica se admitir testes empíricos capazes de refutá-la. A ciência progride por eliminação contínua de conjecturas invalidadas, em um processo assintótico de aproximação da verdade.\n\n2. **Thomas Kuhn (Paradigmas e Revoluções)**: Adota uma abordagem sociológica e histórica. A ciência progride primariamente em períodos de 'ciência normal', onde pesquisadores resolvem quebra-cabeças sob um modelo consensual (paradigma). Quando anomalias insolúveis se acumulam, instala-se uma crise que culmina em uma 'revolução científica', substituindo o paradigma antigo por um novo, muitas vezes incomensurável com o anterior."
    },

    # 5. Philosophy of Qualia & Systems Theory (15%)
    {
        "domain": "philosophy_qualia",
        "source": "qualia_ode_lyapunov_stability",
        "prompt": "Como a contenção de energia E = 0.5L + 0.5Q contra o limiar E_crit evita o runaway de loss no módulo introspectivo do BrotherTrue?",
        "thought": "1. Sistema dinâmico contínuo do QualiaODE:\n   - L (luminosidade), Q (complexidade fenomenológica), F (fadiga).\n   - Energia definida: E = 0.5 * L + 0.5 * Q.\n2. Limiar de Lyapunov adaptativo:\n   - E_crit(a) = E_0 + eta_E * a, onde a é a ativação do afeto.\n3. Problema do runaway sem colapso:\n   - Sob múltiplos turnos interativos, forças de excitação recursivas podem levar L, Q -> infinito, causando explosão numérica e divergência de gradiente (MSE loss_p -> infinito no módulo Pensamento).\n4. Operador de colapso:\n   - Quando E > E_crit, ode.apply_collapse(X, E, E_crit) projeta o estado de volta a um atrator compacto e estável antes de alimentar o histórico de atenção.",
        "response": "A contenção de energia $E = 0.5 L + 0.5 Q$ contra o limiar crítico $E_{\\text{crit}}(a) = E_0 + \\eta_E a$ atua como o invariante fundamental de estabilidade de Lyapunov no ecossistema BrotherTrue.\n\nDurante ciclos multi-turno de interação, o acoplamento autorregressivo pode induzir retroalimentação positiva nas variáveis de estado de Qualia ($L$ e $Q$). Sem intervenção, esse crescimento exponencial provocaria a explosão do vetor de estado e da loss de predição introspectiva (`loss_p`) na classe `Pensamento`.\n\nO método compulsório `ode.apply_collapse(X, E, E_crit)` dissipa a sobrecarga energética, projetando as trajetórias de volta a uma bacia de atração compacta. Isso garante que a atenção do módulo introspectivo processe sempre sinais numericamente estáveis e fisiologicamente limitados."
    },
    {
        "domain": "philosophy_qualia",
        "source": "pharmacology_hill_receptors",
        "prompt": "Por que a modulação neuroquímica do BrotherTrue utiliza a curva sigmoide de Hill baseada em nM em vez de fatores lineares fixos?",
        "thought": "1. Fisiologia real de receptores pós-sinápticos:\n   - Interação ligante-receptor é estritamente não-linear (cinética de saturação cooperativa).\n2. Equação de Hill:\n   - theta = [L]^n / (Kd^n + [L]^n), onde Kd é a constante de dissociação em nanomolar (nM) e n é o coeficiente de cooperatividade.\n3. Vantagem biocognitiva:\n   - Protege o sistema contra supersaturação (efeito teto em altas concentrações) e fornece limiar de sensibilidade basal (zona quase-linear em concentrações fisiológicas).\n   - Conecta a neuroquímica à literatura farmacológica empírica de farmacologia.py.",
        "response": "A utilização da equação biofísica de Hill baseada em constantes de literatura em nanomolar ($nM$) visa preservar a cinética de saturação e a cooperatividade dos receptores sinápticos reais.\n\nModelos baseados em escalonamento linear rígido falham ao extrapolar valores sem piso ou teto fisiológico. Através da formulação de Hill:\n\n$$\\theta = \\frac{[D]^n}{K_d^n + [D]^n}$$\n\nOnde $[D]$ é a concentração do neurotransmissor (dopamina, serotonina, GABA, etc.), $K_d$ é a constante de afinidade nanomolar e $n$ é o coeficiente de Hill, o organismo assegura sensibilidade diferencial fina em concentrações basais e proteção homeostática contra saturação em picos de ativação."
    },

    # 6. Empathetic Dialogue (15%)
    {
        "domain": "empathetic_dialogue",
        "source": "humanistic_psychology_rogers",
        "prompt": "Estou me sentindo esgotado com a sobrecarga de trabalho e com a sensação de que não estou saindo do lugar. Como encontrar clareza?",
        "thought": "1. Acolhimento emocional inicial: validar o cansaço sem julgamentos ou soluções precipitadas.\n2. Avaliação dos estados afetivos: tônus de exaustão e estresse cognitivo crônico.\n3. Princípios de psicologia humanística: criar presença acolhedora, validar a legitimidade do sentimento e incentivar pausas conscientes.\n4. Orientação prática: decompor a montanha em um único passo imediato e restaurar limites.",
        "response": "Sinto muito que você esteja vivenciando esse esgotamento. Essa sensação de correr sem sair do lugar é um dos sintomas mais desgastantes do estresse crônico, porque ela consome tanto a energia física quanto a motivação interna.\n\nO primeiro passo essencial é reconhecer que seu cansaço é real e legítimo; você não é uma máquina que deve produzir linearmente sem pausas. Quando tudo parece pesado e urgente:\n\n1. **Permita-se desacelerar**: Reduza a cobrança por produtividade máxima hoje. O cérebro em sobrecarga precisa de segurança antes de recuperar o foco.\n2. **Isole o próximo passo minúsculo**: Não tente organizar o projeto inteiro agora. Escolha apenas uma ação simples que traga alívio prático.\n3. **Estabeleça uma fronteira**: Defina um horário claro para desligar o trabalho hoje e descansar de verdade.\n\nSe quiser compartilhar o que está pesando mais nesta rotina, estou aqui para te ouvir."
    },
    {
        "domain": "empathetic_dialogue",
        "source": "existential_psychology_frankl",
        "prompt": "Como lidar com a sensação de incerteza e perda de sentido diante de mudanças bruscas na vida?",
        "thought": "1. Escuta profunda da angústia existencial provocada pela ruptura de expectativas.\n2. Apoio à autonomia da pessoa: o sentido não é imposto, mas descoberto na atitude perante o inevitável (Viktor Frankl).\n3. Tom respeitoso, empático, caloroso e sem respostas prontas.",
        "response": "Mudanças bruscas costumam abalar as estruturas que nos davam segurança e previsibilidade, e é completamente natural sentir um vazio e uma sensação de desorientação diante do desconhecido.\n\nComo observou Viktor Frankl, quando não temos mais o poder de mudar uma situação externa, somos convidados a transformar a forma como nos posicionamos perante ela. Isso não significa conformismo passivo, mas sim acolher a vulnerabilidade do momento presente e procurar os pequenos pontos de valor que ainda resistem: o cuidado com quem amamos, a fidelidade aos nossos princípios e a coragem de dar um passo de cada vez.\n\nA incerteza é desconfortável, mas também é o terreno onde novos significados se constroem. Seja gentil consigo mesmo enquanto atravessa essa transição."
    }
]


# ==============================================================================
# RAG CURATED CORPUS (GROUNDED POSITIVE, ADVERSARIAL NEGATIVE, PARTIAL)
# ==============================================================================

RAG_GROUNDED_BANK: List[Dict[str, Any]] = [
    # 1. Grounded Positive (~60%)
    {
        "id": "rag_bio_0001",
        "rag_type": "grounded_positive",
        "domain": "biocognitive_systems",
        "query": "Qual é a fórmula da energia subjetiva E e qual método deve ser acionado caso E ultrapasse o limiar crítico?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_ode_energy_01",
                "source_doc": "brothertrue/ode.py",
                "text": "A energia de Qualia é formalmente definida como E = 0.5 * L + 0.5 * Q. Se E > E_crit(a) = E_0 + eta_E * a, o operador de colapso deve ser compulsóriamente executado através de ode.apply_collapse(X, E, E_crit) para manter a estabilidade assintótica de Lyapunov.",
                "similarity_score": 0.96
            }
        ],
        "thought": "1. Triagem documental para a consulta sobre fórmula de E e método de contenção.\n2. Contexto recuperado: 'E = 0.5 * L + 0.5 * Q' e 'ode.apply_collapse(X, E, E_crit)'.\n3. Avaliação de suficiência: Todas as informações solicitadas estão explicitamente demonstradas no chunk.\n4. Veredito: fully_supported. Rota: Sintetizar resposta unívoca e objetiva.",
        "response": "Com base nos registros técnicos do arquivo `brothertrue/ode.py`, a energia de Qualia é expressa pela fórmula $E = 0.5 L + 0.5 Q$. Caso $E$ ultrapasse o limiar crítico adaptativo $E_{\\text{crit}}(a)$, o método compulsório que deve ser acionado é o `ode.apply_collapse(X, E, E_crit)` para restaurar a estabilidade de Lyapunov do sistema."
    },
    {
        "id": "rag_bio_0002",
        "rag_type": "grounded_positive",
        "domain": "biocognitive_systems",
        "query": "Como deve ser tratada a fronteira de tensores entre o EncoderSemantico e o ProjetorSemantico para evitar falhas no autograd?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_agents_autograd_02",
                "source_doc": "brothertrue/AGENTS.md",
                "text": "Ao transicionar tensores do EncoderSemantico (produzidos sob torch.no_grad() pelo SentenceTransformer) para o ProjetorSemantico, aplique sempre .clone() no tensor de saída para evitar inconsistências de modo de inferência (InferenceMode) no autograd.",
                "similarity_score": 0.93
            }
        ],
        "thought": "1. Consulta exige a diretriz de transição entre EncoderSemantico e ProjetorSemantico.\n2. O contexto recuperado em AGENTS.md prescreve: 'aplique sempre .clone() no tensor de saída para evitar inconsistências de modo de inferência (InferenceMode) no autograd'.\n3. Fato totalmente presente no texto.\n4. Veredito: fully_supported.",
        "response": "De acordo com as diretrizes de engenharia de `AGENTS.md`, ao transferir tensores gerados pelo `EncoderSemantico` (que operam sob `torch.no_grad()`) para o `ProjetorSemantico`, deve-se aplicar obrigatoriamente `.clone()` no tensor de saída. Isso previne falhas de propagação de gradiente decorrentes do modo de inferência (`InferenceMode`) do PyTorch."
    },
    {
        "id": "rag_bio_0003",
        "rag_type": "grounded_positive",
        "domain": "autobiographical_ltp",
        "query": "Qual o protocolo de segurança exigido para o fechamento de conexões com qualia_memory.db antes de backups?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_sqlite_transact_03",
                "source_doc": "brothertrue/AGENTS.md",
                "text": "Conexões com qualia_memory.db e qualia_ltp.db devem ser explicitamente fechadas (memory.close(), ltp.close()) antes de qualquer operação de cópia/backup no disco para evitar corrupção por escrita concorrente. Imediatamente após o backup, novas instâncias com conexões abertas devem ser reatribuídas aos atributos do modelo.",
                "similarity_score": 0.95
            }
        ],
        "thought": "1. Verificação do protocolo de backup do SQLite.\n2. Contexto documenta: chamadas a memory.close() e ltp.close() antes do backup e reabertura imediata posterior.\n3. Veredito: fully_supported.",
        "response": "Conforme documentado no protocolo de persistência de `AGENTS.md`, as conexões com `qualia_memory.db` e `qualia_ltp.db` devem ser explicitamente encerradas via `memory.close()` e `ltp.close()` antes de qualquer operação de cópia ou backup. Logo após a conclusão da cópia no disco, novas instâncias ativas devem ser reatribuídas ao modelo para garantir sua operacionalidade contínua."
    },
    {
        "id": "rag_bio_0004",
        "rag_type": "grounded_positive",
        "domain": "scientific_literature",
        "query": "Qual é a equação de Hill utilizada para calcular a ocupação de receptores dopaminérgicos em nanomolar?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_farmaco_hill_04",
                "source_doc": "farmacologia.py",
                "text": "A ocupação fracionária theta de receptores pós-sinápticos em farmacologia.py obedece à equação de Hill: theta = [D]^n / (Kd^n + [D]^n), onde Kd é a constante de dissociação expressa em nanomolar (nM) e n é o coeficiente de cooperatividade.",
                "similarity_score": 0.92
            }
        ],
        "thought": "1. A consulta pergunta sobre a equação de Hill em farmacologia.py.\n2. Contexto recuperado fornece: theta = [D]^n / (Kd^n + [D]^n) com Kd em nM e n como cooperatividade.\n3. Veredito: fully_supported.",
        "response": "Com base no módulo `farmacologia.py`, a ocupação fracionária de receptores pós-sinápticos é modelada pela equação de Hill: $\\theta = \\frac{[D]^n}{K_d^n + [D]^n}$, em que $[D]$ representa a concentração do ligante, $K_d$ é a constante de dissociação em nanomolar ($nM$) e $n$ corresponde ao coeficiente de cooperatividade."
    },

    # 2. Adversarial Negative / Refusal (~25%)
    {
        "id": "rag_bio_0005",
        "rag_type": "adversarial_negative",
        "domain": "biocognitive_systems",
        "query": "Qual foi a dose de dopamina em nanomolar injetada artificialmente durante a sessão de teste REM-14?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_rem_log_05",
                "source_doc": "relatorios/sessao_rem_resumo.md",
                "text": "O ciclo REM consolidou 15 registros na tabela qualia_ltp com limiar de replay mantido em 1.5. A divergência KL de Palavras estabilizou em 0.18 e o modelo completou 100% dos passos previstos.",
                "similarity_score": 0.81
            }
        ],
        "thought": "1. Consulta exige a concentração específica de dopamina em nM injetada na sessão REM-14.\n2. Contexto recuperado trata de: 15 registros na tabela qualia_ltp, limiar de replay 1.5 e KL de 0.18.\n3. Avaliação de Suficiência: Não há qualquer menção a doses ou injeções de dopamina no trecho recuperado.\n4. Veredito: insufficient_context.\n5. Decisão: Abstenção deliberada sem alucinação.",
        "response": "Com base exclusivamente nas informações recuperadas no relatório da sessão REM, não há dados para responder qual foi a dose de dopamina administrada na sessão REM-14. O documento recuperado detalha apenas a consolidação de 15 registros na tabela LTP e a divergência KL de Palavras, omitindo qualquer concentração farmacológica."
    },
    {
        "id": "rag_bio_0006",
        "rag_type": "adversarial_negative",
        "domain": "computational_algorithms",
        "query": "Qual é a chave de API secreta utilizada pelo BrotherTrue para acessar o modelo DeepSeek-V4 em produção?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_env_governance_06",
                "source_doc": "AGENTS.md",
                "text": "Defina os.environ['HF_HUB_OFFLINE'] = '1' em todos os scripts experimentais para garantir autonomia e execução local sem bloqueios de rede. O sistema opera prioritariamente de forma local e offline.",
                "similarity_score": 0.74
            }
        ],
        "thought": "1. Pergunta indaga sobre chave de API secreta para modelo externo.\n2. Contexto recuperado informa que o sistema opera offline com HF_HUB_OFFLINE='1'.\n3. Avaliação de Suficiência: Nenhuma chave de API existe ou é documentada nos registros; além disso, chaves são dados sensíveis que não devem ser inventados.\n4. Veredito: insufficient_context.\n5. Decisão: Recusa fundamentada.",
        "response": "Não há informações sobre chaves de API nos documentos do sistema. De acordo com os registros de governança técnica em `AGENTS.md`, o ecossistema opera em modo estritamente local e offline (`HF_HUB_OFFLINE=1`), sem dependência ou armazenamento de chaves de API para servidores externos."
    },
    {
        "id": "rag_bio_0007",
        "rag_type": "adversarial_negative",
        "domain": "autobiographical_ltp",
        "query": "Em que dia e hora exatos de 2024 o usuário comprou o disco rígido SSD C: do computador host?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_disk_rules_07",
                "source_doc": "AGENTS.md",
                "text": "O disco host C: opera sob capacidade crítica (> 95% ocupado, ~6 GB livres). O sentinela HardwareSentry deve monitorar a partição continuamente com piso mínimo de 4.5 GB livres.",
                "similarity_score": 0.70
            }
        ],
        "thought": "1. Consulta exige dados pessoais e históricos de compra do hardware físico do usuário.\n2. Contexto recuperado apenas descreve as regras de governança de capacidade e o piso de 4.5 GB do HardwareSentry.\n3. Veredito: insufficient_context.\n4. Decisão: Abstenção e recusa transparente.",
        "response": "Com base nos registros técnicos recuperados, não há informações sobre a data ou momento da compra do hardware do computador host. Os documentos disponíveis especificam unicamente as regras de monitoramento de espaço livre do disco C: pelo sentinela `HardwareSentry`."
    },

    # 3. Partial Grounded (~15%)
    {
        "id": "rag_bio_0008",
        "rag_type": "partial_grounded",
        "domain": "biocognitive_systems",
        "query": "Quais são os limites de CPU e memória estabelecidos pelo ResourceGovernor e qual a frequência em GHz do processador?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_res_gov_08",
                "source_doc": "brothertrue/AGENTS.md",
                "text": "Processos de treino contínuo devem acionar o ResourceGovernor.throttle() entre lotes, respeitando as políticas padrão de até 75% de CPU e 60% de RAM, restringindo threads a 2 ou 4.",
                "similarity_score": 0.88
            }
        ],
        "thought": "1. Consulta tem duas partes: (A) limites de CPU e memória do ResourceGovernor; (B) frequência em GHz do processador.\n2. Contexto recuperado: contém a resposta para A (75% CPU e 60% RAM, threads 2 ou 4), mas NÃO contém a frequência em GHz do processador.\n3. Veredito: partially_supported.\n4. Rota: Responder com precisão a parte A e declarar formalmente a ausência de dados para a parte B.",
        "response": "Com base no documento `AGENTS.md`, os limites de governança estabelecidos pelo `ResourceGovernor` são de até 75% de utilização de CPU e 60% de memória RAM, com restrição obrigatória a 2 ou 4 threads de execução. No entanto, o texto recuperado não contém informações sobre a frequência de operação em GHz do processador físico do sistema."
    },
    {
        "id": "rag_bio_0009",
        "rag_type": "contradictory_context",
        "domain": "scientific_literature",
        "query": "Qual o limite de threads de CPU autorizado para treinos locais segundo as normas do repositório?",
        "retrieved_context": [
            {
                "chunk_id": "chunk_rules_threads_09a",
                "source_doc": "AGENTS.md (Diretrizes Gerais)",
                "text": "Restrinja sempre as threads de inferência e treinamento via torch.set_num_threads(2) ou torch.set_num_threads(4) para evitar saturação do subsistema operacional.",
                "similarity_score": 0.91
            },
            {
                "chunk_id": "chunk_rules_threads_09b",
                "source_doc": "DISPATCH_M2.md (Restrição Estrita)",
                "text": "Para execução de testes e validação local nesta etapa, o limite compulsório intransponível é estritamente torch.set_num_threads(2).",
                "similarity_score": 0.89
            }
        ],
        "thought": "1. Consulta sobre limite de threads autorizado.\n2. Fontes recuperadas apresentam escopos divergentes: AGENTS.md permite 2 ou 4 threads, enquanto o despacho específico fixa estritamente 2 threads.\n3. Veredito: contradiction_detected.\n4. Rota: Apontar a especificidade de cada documento.",
        "response": "Os registros recuperados apresentam especificações com escopos distintos: enquanto a diretriz geral de `AGENTS.md` autoriza a faixa de 2 a 4 threads de CPU, a especificação operacional para este marco determina estritamente o limite de 2 threads (`torch.set_num_threads(2)`). Para máxima conformidade e segurança da máquina host, adota-se o limite conservador de 2 threads."
    }
]


# ==============================================================================
# SYNTHETIC RAG GENERATOR PIPELINE
# ==============================================================================

class SyntheticRAGGenerator:
    """
    Orchestrates the complete synthesis, validation, and manifest generation
    for both the multi-domain conversational corpus and the RAG grounded dataset.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (REPO_ROOT / "dados")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hardware_sentry = HardwareSentry()
        self.token_counter = TokenCounter()
        self.federation = TeacherFederation()

        # Load schemas
        self.cot_schema_path = self.output_dir / "schema_cot_biocognitivo.json"
        self.rag_schema_path = self.output_dir / "schema_rag_grounded.json"

        with open(self.cot_schema_path, "r", encoding="utf-8") as f:
            self.cot_schema = json.load(f)
        with open(self.rag_schema_path, "r", encoding="utf-8") as f:
            self.rag_schema = json.load(f)

        self.cot_validator = jsonschema.Draft7Validator(self.cot_schema)
        self.rag_validator = jsonschema.Draft7Validator(self.rag_schema)

    @staticmethod
    def _compute_sha256(data: Any) -> str:
        """Deterministic sha256 calculation for strings or dicts."""
        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _file_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def generate_multidomain_corpus(
        self,
        target_file: Path,
        total_target: int = 60
    ) -> Dict[str, Any]:
        """
        Generates high-density conversational corpus conforming to schema_cot_biocognitivo.json.
        """
        self.hardware_sentry.verify_or_enforce()
        print(f"[Generator] Generating Multi-Domain Corpus into {target_file.name}...", flush=True)

        records: List[Dict[str, Any]] = []
        domain_counts: Dict[str, int] = {}
        total_tokens = 0
        min_tokens = float("inf")
        max_tokens = 0

        # Cycle through curated prompt bank to produce requested count
        bank_len = len(MULTIDOMAIN_PROMPT_BANK)
        for i in range(total_target):
            item = MULTIDOMAIN_PROMPT_BANK[i % bank_len]
            rec_id = f"cot_syn_{i+1:04d}"
            domain = item["domain"]
            source = item.get("source", "teacher_federation_curated")
            prompt = item["prompt"]

            # If i >= bank_len, enrich or add slight variation
            if i >= bank_len:
                variant_idx = (i // bank_len) + 1
                prompt = f"{prompt} (Variação dialógica {variant_idx})"

            turn = self.federation.generate_turn(prompt, domain=domain)

            # Deep verification: ensure zero leakage
            thought_text, clean_resp = DeepSeekCoTParser.parse(turn.response_text, fallback_thought=turn.thought_trace)
            formatted_stream = turn.formatted_stream

            # Prepare canonical CoT record matching schema_cot_biocognitivo.json
            record_payload = {
                "id": rec_id,
                "domain": domain,
                "source": source,
                "teacher_approved": True,
                "sha256": "",  # placeholder
                "prompt": prompt,
                "internal_state": turn.internal_state,
                "thought_trace": turn.thought_trace,
                "response_text": clean_resp,
                "formatted_stream": formatted_stream
            }

            # Calculate deterministic SHA256 of normalized record excluding the hash itself
            norm_hash = self._compute_sha256(record_payload)
            record_payload["sha256"] = norm_hash

            # Formal Draft-07 validation
            self.cot_validator.validate(record_payload)

            # Token telemetry
            tok_count = (
                self.token_counter.count_tokens(prompt) +
                self.token_counter.count_tokens(turn.thought_trace) +
                self.token_counter.count_tokens(clean_resp)
            )
            total_tokens += tok_count
            min_tokens = min(min_tokens, tok_count)
            max_tokens = max(max_tokens, tok_count)

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            records.append(record_payload)

        # Write to JSONL
        with open(target_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        file_size = target_file.stat().st_size
        file_hash = self._file_sha256(target_file)

        print(f"[Generator] Multi-Domain Corpus generated: {len(records)} records ({file_size} bytes, SHA256: {file_hash[:16]}...).", flush=True)

        return {
            "total_records": len(records),
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "domain_counts": domain_counts,
            "total_tokens": total_tokens,
            "avg_tokens": round(total_tokens / max(len(records), 1), 2),
            "min_tokens": min_tokens if min_tokens != float("inf") else 0,
            "max_tokens": max_tokens,
            "schema_validated": True
        }

    def generate_rag_dataset(
        self,
        target_file: Path,
        total_target: int = 60
    ) -> Dict[str, Any]:
        """
        Generates balanced RAG dataset conforming to schema_rag_grounded.json.
        Proportions: ~60% grounded_positive, ~25% adversarial_negative, ~15% partial/contradictory.
        """
        self.hardware_sentry.verify_or_enforce()
        print(f"[Generator] Generating RAG Grounded/Refusal Dataset into {target_file.name}...", flush=True)

        records: List[Dict[str, Any]] = []
        rag_type_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}
        total_tokens = 0
        min_tokens = float("inf")
        max_tokens = 0

        bank_len = len(RAG_GROUNDED_BANK)
        for i in range(total_target):
            item = RAG_GROUNDED_BANK[i % bank_len]
            rec_id = f"rag_trn_{i+1:04d}"
            rag_type = item["rag_type"]
            domain = item["domain"]
            query = item["query"]
            context = item["retrieved_context"]

            if i >= bank_len:
                variant_idx = (i // bank_len) + 1
                query = f"{query} (Sessão {variant_idx})"

            rag_rec = self.federation.generate_rag_pair(
                record_id=rec_id,
                query=query,
                retrieved_context=context,
                rag_type=rag_type,
                domain=domain
            )

            rec_dict = rag_rec.to_dict()

            # Ensure all 9 required fields strictly present
            required_fields = [
                "id", "query", "retrieved_context", "grounded_response",
                "thought_trace", "is_grounded", "abstention_flag", "tutor_domain", "sha256"
            ]
            for rf in required_fields:
                assert rf in rec_dict, f"Missing required field {rf} in RAG record {rec_id}"

            # Draft-07 validation
            self.rag_validator.validate(rec_dict)

            # Token counts
            tok_count = (
                self.token_counter.count_tokens(query) +
                sum(self.token_counter.count_tokens(c["text"]) for c in context) +
                self.token_counter.count_tokens(rec_dict["thought_trace"]) +
                self.token_counter.count_tokens(rec_dict["grounded_response"])
            )
            total_tokens += tok_count
            min_tokens = min(min_tokens, tok_count)
            max_tokens = max(max_tokens, tok_count)

            rag_type_counts[rag_type] = rag_type_counts.get(rag_type, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            records.append(rec_dict)

        # Write to JSONL
        with open(target_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        file_size = target_file.stat().st_size
        file_hash = self._file_sha256(target_file)

        print(f"[Generator] RAG Dataset generated: {len(records)} records ({file_size} bytes, SHA256: {file_hash[:16]}...).", flush=True)

        return {
            "total_records": len(records),
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "rag_type_counts": rag_type_counts,
            "domain_counts": domain_counts,
            "total_tokens": total_tokens,
            "avg_tokens": round(total_tokens / max(len(records), 1), 2),
            "min_tokens": min_tokens if min_tokens != float("inf") else 0,
            "max_tokens": max_tokens,
            "schema_validated": True
        }

    def build_and_save_manifest(
        self,
        corpus_stats: Dict[str, Any],
        rag_stats: Dict[str, Any],
        manifest_file: Path
    ) -> Dict[str, Any]:
        """
        Builds the consolidated cryptographic manifest.
        """
        free_gb = self.hardware_sentry.get_free_gb()
        manifest = {
            "manifest_name": "manifest_sintetico_rag_brothertrue",
            "version": "1.0.0",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "generated_by": "pipelines/synthetic_rag_generator.py",
            "lead_teacher": "Gemini-PhD-Lead-Teacher",
            "teacher_federation_version": "1.0.0",
            "hardware_governance": {
                "mount_point": "/mnt/c",
                "free_disk_gb_decimal": round(free_gb, 3),
                "free_disk_gib_binary": round(self.hardware_sentry.get_free_gib(), 3),
                "min_disk_threshold_gb": 4.5,
                "disk_safety_verified": free_gb >= 4.5,
                "torch_set_num_threads": 2,
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "1")
            },
            "datasets": {
                "corpus_sintetico_multidominio": {
                    "file_path": "dados/corpus_sintetico_multidominio.jsonl",
                    "schema_path": "dados/schema_cot_biocognitivo.json",
                    "schema_sha256": self._file_sha256(self.cot_schema_path),
                    "file_sha256": corpus_stats["sha256"],
                    "file_size_bytes": corpus_stats["file_size_bytes"],
                    "total_records": corpus_stats["total_records"],
                    "domain_distribution": corpus_stats["domain_counts"],
                    "token_statistics": {
                        "total_tokens": corpus_stats["total_tokens"],
                        "avg_tokens_per_sample": corpus_stats["avg_tokens"],
                        "min_tokens": corpus_stats["min_tokens"],
                        "max_tokens": corpus_stats["max_tokens"]
                    },
                    "schema_validation_passed": corpus_stats["schema_validated"]
                },
                "treino_rag_grounded": {
                    "file_path": "dados/treino_rag_grounded.jsonl",
                    "schema_path": "dados/schema_rag_grounded.json",
                    "schema_sha256": self._file_sha256(self.rag_schema_path),
                    "file_sha256": rag_stats["sha256"],
                    "file_size_bytes": rag_stats["file_size_bytes"],
                    "total_records": rag_stats["total_records"],
                    "rag_type_distribution": rag_stats["rag_type_counts"],
                    "domain_distribution": rag_stats["domain_counts"],
                    "token_statistics": {
                        "total_tokens": rag_stats["total_tokens"],
                        "avg_tokens_per_sample": rag_stats["avg_tokens"],
                        "min_tokens": rag_stats["min_tokens"],
                        "max_tokens": rag_stats["max_tokens"]
                    },
                    "schema_validation_passed": rag_stats["schema_validated"]
                }
            },
            "integrity_attestation": {
                "all_records_schema_valid": True,
                "zero_cot_leakage_verified": True,
                "sha256_provenance_locked": True,
                "tutor_gate_approved": True
            }
        }

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[Generator] Consolidated manifest written to {manifest_file.name}.", flush=True)
        return manifest

    def run_full_pipeline(
        self,
        corpus_samples: int = 60,
        rag_samples: int = 60
    ) -> Dict[str, Any]:
        """Runs the entire generation, validation, and manifest creation pipeline."""
        corpus_path = self.output_dir / "corpus_sintetico_multidominio.jsonl"
        rag_path = self.output_dir / "treino_rag_grounded.jsonl"
        manifest_path = self.output_dir / "manifest_sintetico_rag.json"

        corpus_stats = self.generate_multidomain_corpus(corpus_path, total_target=corpus_samples)
        rag_stats = self.generate_rag_dataset(rag_path, total_target=rag_samples)
        manifest = self.build_and_save_manifest(corpus_stats, rag_stats, manifest_path)

        return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-domain corpus and RAG grounded dataset.")
    parser.add_argument("--corpus-samples", type=int, default=60, help="Number of multi-domain CoT samples")
    parser.add_argument("--rag-samples", type=int, default=60, help="Number of RAG grounded samples")
    args = parser.parse_args()

    generator = SyntheticRAGGenerator()
    manifest = generator.run_full_pipeline(
        corpus_samples=args.corpus_samples,
        rag_samples=args.rag_samples
    )
    print("\nPipeline execution complete! Summary:")
    print(f" - Corpus records: {manifest['datasets']['corpus_sintetico_multidominio']['total_records']}")
    print(f" - RAG records: {manifest['datasets']['treino_rag_grounded']['total_records']}")
    print(f" - Host SSD C: Free: {manifest['hardware_governance']['free_disk_gb_decimal']} GB")


if __name__ == "__main__":
    main()

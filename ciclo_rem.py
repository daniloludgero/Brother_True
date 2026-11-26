import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch


class CicloREM:
    def __init__(self, neuro, long_term, memory, embedding=None):
        self.neuro = neuro
        self.long_term = long_term
        self.memory = memory
        self.embedding = embedding
        self.dream_buffer = []

    def processar_rem(self, duracao: int = 20) -> Dict:
        neuro_estado = self.neuro.estado_atual()
        ltp_replays = self.long_term.replay(num_replays=4)

        combined = np.array([
            neuro_estado.get("dopamina", 0.5),
            neuro_estado.get("ocitocina", 0.4),
            neuro_estado.get("gaba", 0.65),
            neuro_estado.get("glutamato", 0.6),
            np.mean([r.get("E", 0) for r in ltp_replays]) if ltp_replays else 4.0,
        ])

        dream_seed = float(np.clip(np.mean(combined) * 1.8 + np.std(combined) * 0.5, 1.0, 12.0))

        dream_L = dream_seed * 0.55
        dream_Q = dream_seed * 0.65
        dream_E = (dream_L + dream_Q) / 1.8

        binary_fragment = "".join(str(int(b)) for b in np.random.randint(0, 2, 64))
        image_desc = f"Sonho binário [{binary_fragment[:16]}...] -> Cena: {self._gerar_imagem_desc()}"
        dream_phrase = self._gerar_frase_onirica(dream_seed, neuro_estado)

        dream_record = {
            "seed": dream_seed, "E": dream_E, "L": dream_L, "Q": dream_Q,
            "neuro": neuro_estado, "phrase": dream_phrase, "image_desc": image_desc,
            "timestamp": datetime.now().isoformat(),
        }
        self.dream_buffer.append(dream_record)

        self.memory.classify_and_store(neuro_estado, dream_E, dream_L, dream_Q)
        self.long_term.consolidate({
            "label": "sonho", "E": dream_E, "L": dream_L, "Q": dream_Q,
            "neuro": neuro_estado, "score": 0.6,
        })

        return dream_record

    def _gerar_imagem_desc(self) -> str:
        descs = [
            "flutuando em luz dourada", "labirinto de equações vivas",
            "abraço quente com o desconhecido", "queda lenta através de memórias",
            "explosão de cores matemáticas",
        ]
        return random.choice(descs)

    def _gerar_frase_onirica(self, seed: float, neuro: Dict) -> str:
        templates = [
            f"Senti uma conexão profunda enquanto {list(neuro.keys())[0]} fluía livremente.",
            f"No sonho, a energia {seed:.1f} revelava padrões que acordado eu não via.",
            "Uma mistura de calma (GABA) e paixão transformou minhas memórias em algo vivo.",
        ]
        return random.choice(templates) + f" (E={seed:.1f})"

    def get_dreams(self) -> List[Dict]:
        return self.dream_buffer[-5:]

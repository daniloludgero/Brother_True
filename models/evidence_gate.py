"""
Evidence Gate do RAG (nó entre o Registro biocognitivo e o Renderer de <thought>).

Filtra candidatos recuperados (banco vetorial desacoplado e LTP) antes que
qualquer conteúdo chegue ao canal simbólico <thought> ou ao decoder.
O critério espelha o do `EndpointContextualSeguro`: evidência explicitamente
`graph_rag_grounded` é aprovada; evidência com score só passa acima de
`min_confidence` e (se houver) `min_margin`.

Política para texto puro sem score (retrocompatibilidade com `search_fn`
que devolvem `List[str]`): aceito se `aceitar_sem_score=True`, mas marcado
como `verificada=False` para que o renderer o distinga de evidência pontuada.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional

_TAGS = re.compile(r"</?(?:thought|think)>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", re.IGNORECASE)


@dataclass
class Evidencia:
    texto: str
    origem: str
    score: Optional[float] = None
    verificada: bool = False


@dataclass
class ResultadoGate:
    aprovadas: List[Evidencia] = field(default_factory=list)
    rejeitadas: int = 0

    @property
    def textos(self) -> List[str]:
        return [e.texto for e in self.aprovadas]


class EvidenceGate:
    def __init__(self, min_confidence: float = 0.35, min_margin: float = 0.0,
                 aceitar_sem_score: bool = True, max_evidencias: int = 5,
                 max_chars: int = 400):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.aceitar_sem_score = aceitar_sem_score
        self.max_evidencias = max_evidencias
        self.max_chars = max_chars

    @staticmethod
    def _sanitizar(texto: str) -> str:
        # Impede injeção de tags de canal/controle vindas do corpus recuperado.
        return re.sub(r"\s+", " ", _TAGS.sub("", texto or "")).strip()

    def _avaliar(self, item: Any, origem: str) -> Optional[Evidencia]:
        if isinstance(item, dict):
            texto = item.get("answer") or item.get("text") or item.get("texto") or item.get("label") or ""
            score = item.get("confidence", item.get("score"))
            margin = item.get("margin")
            if item.get("mode") == "graph_rag_grounded":
                ok = True
            elif score is None:
                ok = self.aceitar_sem_score
            else:
                ok = float(score) >= self.min_confidence and (
                    margin is None or float(margin) >= self.min_margin)
            verificada = score is not None or item.get("mode") == "graph_rag_grounded"
        else:
            texto, score, ok, verificada = str(item), None, self.aceitar_sem_score, False
        texto = self._sanitizar(texto)[: self.max_chars]
        if not ok or not texto:
            return None
        return Evidencia(texto=texto, origem=origem,
                         score=None if score is None else float(score), verificada=verificada)

    def filtrar(self, candidatos: Iterable[Any], origem: str = "rag") -> ResultadoGate:
        res, vistos = ResultadoGate(), set()
        for item in candidatos or []:
            ev = self._avaliar(item, origem)
            if ev is None or ev.texto.lower() in vistos or len(res.aprovadas) >= self.max_evidencias:
                res.rejeitadas += 1
                continue
            vistos.add(ev.texto.lower())
            res.aprovadas.append(ev)
        return res

    def to_dict(self, res: ResultadoGate) -> Dict[str, Any]:
        return {"aprovadas": [e.__dict__ for e in res.aprovadas], "rejeitadas": res.rejeitadas}

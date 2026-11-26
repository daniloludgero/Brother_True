"""Evidence decisions supplied by an adjudicator, never inferred from retrieval scores."""

from dataclasses import dataclass
import re


_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
VERDICTS = ("supported", "partial", "insufficient", "conflicting")
MODES = ("grounded", "general", "creative")


def validate_id(value: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("IDs must be 1..128 ASCII alphanumeric or _ . : - characters")


@dataclass(frozen=True)
class EvidenceSource:
    id: str
    text: str

    def __post_init__(self) -> None:
        validate_id(self.id)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Evidence sources require nonempty text")


@dataclass(frozen=True)
class Citation:
    source_id: str
    quote: str

    def __post_init__(self) -> None:
        validate_id(self.source_id)
        if not isinstance(self.quote, str) or not self.quote.strip():
            raise ValueError("Citations require a nonempty exact quotation")


@dataclass(frozen=True)
class EvidenceClaim:
    id: str
    subquestion: str
    verdict: str
    citations: tuple[Citation, ...] = ()
    adjudicator: str = ""
    adjudication_id: str = ""

    def __post_init__(self) -> None:
        validate_id(self.id)
        if not isinstance(self.subquestion, str) or not self.subquestion.strip():
            raise ValueError("A subquestion is required")
        if self.verdict not in VERDICTS:
            raise ValueError("Unknown evidence verdict")
        object.__setattr__(self, "citations", tuple(self.citations))
        if any(not isinstance(c, Citation) for c in self.citations):
            raise TypeError("citations must contain Citation objects")
        # An unreviewed claim is always insufficient, even if relevant text exists.
        if self.verdict != "insufficient":
            if (not isinstance(self.adjudicator, str) or not self.adjudicator.strip()
                    or not isinstance(self.adjudication_id, str) or not self.adjudication_id.strip()):
                raise ValueError("Supported/partial/conflicting claims require adjudication provenance")
            if not self.citations:
                raise ValueError("Reviewed non-insufficient verdicts require citations")
        if self.verdict == "conflicting" and len(self.citations) < 2:
            raise ValueError("A conflict needs at least two cited passages")


@dataclass(frozen=True)
class EvidenceDecision:
    mode: str
    claims: tuple[EvidenceClaim, ...] = ()
    sources: tuple[EvidenceSource, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("Unknown response mode")
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "sources", tuple(self.sources))
        if any(not isinstance(c, EvidenceClaim) for c in self.claims):
            raise TypeError("claims must contain EvidenceClaim objects")
        if any(not isinstance(s, EvidenceSource) for s in self.sources):
            raise TypeError("sources must contain EvidenceSource objects")
        if len({c.id for c in self.claims}) != len(self.claims):
            raise ValueError("Duplicate subquestion IDs")
        sources = {s.id: s.text for s in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("Duplicate source IDs")
        for claim in self.claims:
            for citation in claim.citations:
                if citation.source_id not in sources:
                    raise ValueError("Citation source was not supplied")
                if citation.quote not in sources[citation.source_id]:
                    raise ValueError("Citation quote must occur exactly in its source")

    @property
    def verdict(self) -> str:
        """Aggregate supplied judgments conservatively; this is not an entailment model."""
        verdicts = {c.verdict for c in self.claims}
        if not verdicts:
            return "insufficient"
        if "conflicting" in verdicts:
            return "conflicting"
        if verdicts == {"supported"}:
            return "supported"
        if verdicts == {"insufficient"}:
            return "insufficient"
        return "partial"

    @property
    def directive(self) -> str:
        if self.mode == "creative":
            return "Inventar ou abstrair é permitido; identificar a saída como ficção ou hipótese, sem atribuição factual às fontes."
        if self.mode == "general":
            return "Conhecimento paramétrico e abstração são permitidos; distinguir inferências de fatos documentados e declarar incerteza."
        return {
            "supported": "Responder somente o que as passagens revisadas sustentam, citando as fontes.",
            "partial": "Responder apenas as partes sustentadas e declarar quais subquestões permanecem sem suporte.",
            "insufficient": "Declarar a falta de evidência documental; pedir contexto ou propor hipótese explicitamente marcada, sem inventar fatos.",
            "conflicting": "Apresentar a divergência entre fontes; não escolher uma conclusão factual sem justificativa adicional.",
        }[self.verdict]

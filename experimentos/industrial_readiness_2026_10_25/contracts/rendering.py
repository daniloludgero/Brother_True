"""Deterministic conditioning and a separate audit channel for the own decoder."""

from dataclasses import dataclass
import json
import re

from .evidence import EvidenceDecision
from .state import BiocognitiveState


class ThoughtRenderer:
    @staticmethod
    def render(state: BiocognitiveState, decision: EvidenceDecision) -> str:
        # No retrieved prose, subquestion text, free-form adjudicator or source
        # metadata enters the control channel. Retrieved passages remain data.
        content = {
            "kind": "biocognitive_projection_v1",
            "not_a_generated_reasoning_trace": True,
            "vector20": list(state.vector20),
            "energy": state.energy,
            "energy_crit": state.energy_crit,
            "mode": decision.mode,
            "evidence_verdict": decision.verdict,
            "claims": [
                {"id": c.id, "verdict": c.verdict,
                 "source_ids": sorted({q.source_id for q in c.citations})}
                for c in decision.claims
            ],
            "directive": decision.directive,
        }
        return "<thought>\n" + json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n</thought>"

    @staticmethod
    def summary(state: BiocognitiveState, decision: EvidenceDecision) -> str:
        return ("Projeção auditável do estado biocognitivo; "
                f"modo={decision.mode}; evidência={decision.verdict}; "
                f"subquestões={len(decision.claims)}. {decision.directive}")


@dataclass(frozen=True)
class ResponseDiagnostic:
    conditioning_projection: str
    summary: str
    output_status: str
    removed_thought_blocks: int


@dataclass(frozen=True)
class ResponseEnvelope:
    response: str
    diagnostic: ResponseDiagnostic


_TAG = re.compile(r"</?(thought|think)>", re.IGNORECASE)
_SUSPICIOUS = re.compile(r"</?(?:thought|think)\b[^>]*>|</?(?:thought|think)[^>]*$", re.IGNORECASE)
_PARTIAL_TAIL = re.compile(r"</?(?:t|th|tho|thou|thoug|though|thought|thi|thin|think)$", re.IGNORECASE)
_CONTROL = re.compile(r"<\|[^>]*\|>|<(?:bos|eos|pad)>", re.IGNORECASE)
_CONTROL_TAIL = re.compile(r"<\|[^>]*$", re.IGNORECASE)


def _clean(raw: str) -> tuple[str, str, int]:
    # On malformed/truncated channel boundaries the entire response is withheld.
    # This avoids guessing which suffix belongs to the public verbal channel.
    if _PARTIAL_TAIL.search(raw):
        return "", "blocked_channel_boundary", 0
    valid_spans = {m.span() for m in _TAG.finditer(raw)}
    if any(m.span() not in valid_spans for m in _SUSPICIOUS.finditer(raw)):
        return "", "blocked_channel_boundary", 0
    stack: list[str] = []
    public: list[str] = []
    position = 0
    removed = 0
    for match in _TAG.finditer(raw):
        if not stack:
            public.append(raw[position:match.start()])
        name = match.group(1).lower()
        if raw[match.start() + 1] == "/":
            if not stack or stack[-1] != name:
                return "", "blocked_channel_boundary", removed
            stack.pop()
            if not stack:
                removed += 1
        else:
            stack.append(name)
        position = match.end()
    if stack:
        return "", "blocked_channel_boundary", removed
    public.append(raw[position:])
    response = "".join(public).strip()
    if _CONTROL.search(response) or _CONTROL_TAIL.search(response):
        return "", "blocked_control_token", removed
    return response, "clean" if response else "empty", removed


def build_response(raw: str, state: BiocognitiveState,
                   decision: EvidenceDecision) -> ResponseEnvelope:
    """Expose one generated response plus an explicitly separate diagnostic.

    Channel filtering does not establish semantic truth or evidence compliance.
    Those are measured by independent evaluation and/or an explicit adjudicator.
    Buffer before publication: this function does not authorize raw token streaming.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be a generated string")
    if len(raw) > 65536:
        raise ValueError("Generated response exceeds the bounded channel parser")
    response, status, removed = _clean(raw)
    return ResponseEnvelope(response, ResponseDiagnostic(
        ThoughtRenderer.render(state, decision), ThoughtRenderer.summary(state, decision),
        status, removed,
    ))

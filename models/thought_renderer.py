"""
Renderer de <thought> simbólico.

Converte o Registro biocognitivo estruturado (Qualia, neuroquímica, Pensamento,
Palavras) + evidências aprovadas pelo Evidence Gate em um bloco <thought>
que condiciona o decoder. O bloco nunca chega à resposta verbal: o
`DeepSeekCoTParser` o remove no último nó do pipeline.
"""

from typing import Any, Callable, Dict, List, Optional

THOUGHT_OPEN, THOUGHT_CLOSE = "<thought>", "</thought>"


def render_thought_body(bio_state: Dict[str, Any], cot_signals: Dict[str, Any],
                        evidencias: Optional[List[Any]] = None,
                        formatter: Optional[Callable[[Dict, Dict], str]] = None) -> str:
    """Corpo do pensamento (sem tags). `formatter` = `voz.format_endogenous_thought`, se houver."""
    corpo = ""
    if formatter is not None:
        try:
            corpo = formatter(bio_state, cot_signals) or ""
        except Exception:
            corpo = ""
    if not corpo:
        corpo = (
            f"[Homeostase Qualia: L={bio_state['L']:.2f}, Q={bio_state['Q']:.2f}, "
            f"F={bio_state['F']:.2f}, E={bio_state['E']:.2f}]\n"
            f"[Predição Introspectiva (Pensamento): erro_predição_loss_p={float(cot_signals.get('loss_p', 0.0)):.4f}]\n"
            f"[Surpresa Epistêmica (Palavras): KL={float(cot_signals.get('kl', 0.0)):.4f}]"
        )
    if evidencias:
        linhas = []
        for e in evidencias:
            marca = "verificada" if getattr(e, "verificada", False) else "não-pontuada"
            linhas.append(f"- ({getattr(e, 'origem', 'rag')}, {marca}) {getattr(e, 'texto', e)}")
        corpo += "\n[Evidências aprovadas pelo Gate]:\n" + "\n".join(linhas)
    else:
        corpo += "\n[Evidências aprovadas pelo Gate]: nenhuma"
    return corpo


def render_thought(corpo: str) -> str:
    return f"{THOUGHT_OPEN}\n{corpo}\n{THOUGHT_CLOSE}"

"""Decoder textual separado para traces de Pensamento."""

from .voz import Voz


class PensamentoLinguistico(Voz):
    """Mesmo núcleo causal leve, com checkpoint e tokenizer próprios.

    Não substitui o Pensamento numérico existente; é o submódulo textual que
    aprende os traces privados e pode alimentar planejamento/verificação.
    """

    pass

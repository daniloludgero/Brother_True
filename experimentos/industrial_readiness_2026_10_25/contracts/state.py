"""Validated snapshot of the canonical core vector; no physiological defaults."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable


@dataclass(frozen=True)
class BiocognitiveState:
    """20D core layout: X3, receptors8, prediction3, delta3, loss, KL, surprise.

    The caller derives receptors through farmacologia/Hill, invokes collapse before
    Pensamento/history, and passes a detached CPU snapshot. Validation checks the
    numeric result, not the authenticity of the caller's provenance declaration.
    """

    vector20: tuple[float, ...]
    source: str
    energy_crit: float
    collapse_checked: bool

    def __post_init__(self) -> None:
        values = tuple(self.vector20)
        if len(values) != 20:
            raise ValueError("A canonical core snapshot requires exactly 20 values")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
            raise ValueError("Core values must be finite real numbers")
        values = tuple(float(v) for v in values)
        object.__setattr__(self, "vector20", values)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Core provenance is required")
        if isinstance(self.energy_crit, bool) or not isinstance(self.energy_crit, Real):
            raise ValueError("energy_crit must be a finite positive real")
        if not math.isfinite(self.energy_crit) or self.energy_crit <= 0:
            raise ValueError("energy_crit must be a finite positive real")
        if self.collapse_checked is not True:
            raise ValueError("The core must evaluate collapse before exporting state")
        if any(not 0 <= v <= 1 for v in values[3:11]):
            raise ValueError("Receptor occupancies must be in [0,1]")
        for current, predicted, delta in zip(values[:3], values[11:14], values[14:17]):
            if not math.isclose(current - predicted, delta, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Delta must equal current minus predicted state")
        if values[17] < 0 or values[18] < 0 or values[19] not in (0.0, 1.0):
            raise ValueError("Loss/KL must be nonnegative and surprise must be binary")
        if self.energy > float(self.energy_crit) + 1e-6:
            raise ValueError("Exported energy remains above energy_crit after collapse check")

    @classmethod
    def from_core(cls, vector20: Iterable[float], *, source: str,
                  energy_crit: float, collapse_checked: bool) -> "BiocognitiveState":
        return cls(tuple(vector20), source, energy_crit, collapse_checked)

    @property
    def energy(self) -> float:
        return 0.5 * self.vector20[0] + 0.5 * self.vector20[1]

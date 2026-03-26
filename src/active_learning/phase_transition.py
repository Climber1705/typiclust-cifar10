import numpy as np
from typing import Optional


class PhaseTransitionEstimator:

    def __init__(self, n_classes: int, default_factor: float = 5.0):
        self._default = default_factor * n_classes
        self._history = []
        self._m0: Optional[float] = None

    def update(self, cumulative_budget: int, validation_accuracy: float) -> None:
        self._history.append((cumulative_budget, validation_accuracy))
        if len(self._history) < 3:
            return

        budgets = np.array([h[0] for h in self._history], dtype=float)
        accs    = np.array([h[1] for h in self._history], dtype=float)
        gains   = np.diff(accs) / np.diff(budgets)

        falling = np.where(np.diff(gains) < 0)[0]
        if len(falling) > 0:
            self._m0 = float(budgets[falling[0] + 1])

    @property
    def m0(self) -> float:
        return self._m0 if self._m0 is not None else self._default

def compute_lambda(cumulative_budget: int, m0: float, steepness: float = 0.1) -> float:
    exponent = np.clip(steepness * (cumulative_budget - m0), -50, 50)
    return 1.0 / (1.0 + np.exp(exponent))
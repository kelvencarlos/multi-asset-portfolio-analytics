from __future__ import annotations

import numpy as np


def stress_test(weights: np.ndarray, shocks: np.ndarray) -> float:
    """Retorna o impacto agregado do cenário de estresse."""
    clean_weights = np.asarray(weights, dtype=float)
    clean_shocks = np.asarray(shocks, dtype=float)
    if clean_weights.shape != clean_shocks.shape:
        raise ValueError("Vetores de pesos e choques devem possuir o mesmo tamanho.")
    return float((clean_weights * clean_shocks).sum())

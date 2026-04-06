from __future__ import annotations

import numpy as np
import pandas as pd


def volatility(port_return: pd.Series) -> float:
    """Volatilidade anualizada assumindo 252 pregões."""
    if port_return.empty:
        return 0.0
    return float(port_return.std() * np.sqrt(252))


def drawdown(cumulative: pd.Series) -> pd.Series:
    """Drawdown relativo ao último pico histórico."""
    if cumulative.empty:
        return pd.Series(dtype=float)
    peak = cumulative.cummax()
    return (cumulative / peak) - 1


def risk_contribution(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Contribuição marginal de risco por ativo."""
    if returns.empty:
        return np.zeros_like(weights, dtype=float)

    cov = returns.cov()
    clean_weights = np.asarray(weights, dtype=float)
    portfolio_vol = np.sqrt(clean_weights.T @ cov @ clean_weights)

    if not np.isfinite(portfolio_vol) or portfolio_vol <= 0:
        return np.zeros_like(clean_weights, dtype=float)

    contrib = (clean_weights * (cov @ clean_weights)) / portfolio_vol
    return np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)

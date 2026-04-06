from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos simples diários da série de preços."""
    if data.empty:
        return pd.DataFrame()
    return data.pct_change().dropna(how="all")


def portfolio_return(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Calcula retorno agregado da carteira com pesos fixos."""
    if returns.empty:
        return pd.Series(dtype=float)
    clean_weights = np.asarray(weights, dtype=float)
    return (returns * clean_weights).sum(axis=1)


def cumulative_return(port_return: pd.Series) -> pd.Series:
    """Converte retorno diário em trajetória acumulada."""
    if port_return.empty:
        return pd.Series(dtype=float)
    return (1 + port_return).cumprod()

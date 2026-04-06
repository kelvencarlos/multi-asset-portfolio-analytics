import numpy as np

def volatility(port_return):
    return port_return.std() * np.sqrt(252)

def drawdown(cumulative):
    peak = cumulative.cummax()
    return (cumulative / peak) - 1

def risk_contribution(returns, weights):
    cov = returns.cov()
    portfolio_vol = np.sqrt(weights.T @ cov @ weights)

    if not np.isfinite(portfolio_vol) or portfolio_vol <= 0:
        return np.zeros_like(weights, dtype=float)

    contrib = (weights * (cov @ weights)) / portfolio_vol

    return np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
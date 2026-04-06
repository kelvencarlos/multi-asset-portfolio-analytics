import numpy as np

def volatility(port_return):
    return port_return.std() * np.sqrt(252)

def drawdown(cumulative):
    peak = cumulative.cummax()
    return (cumulative / peak) - 1

def risk_contribution(returns, weights):
    cov = returns.cov()
    portfolio_vol = np.sqrt(weights.T @ cov @ weights)

    contrib = (weights * (cov @ weights)) / portfolio_vol

    return contrib
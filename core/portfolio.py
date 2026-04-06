import numpy as np

def calculate_returns(data):
    return data.pct_change().dropna()

def portfolio_return(returns, weights):
    return (returns * weights).sum(axis=1)

def cumulative_return(port_return):
    return (1 + port_return).cumprod()
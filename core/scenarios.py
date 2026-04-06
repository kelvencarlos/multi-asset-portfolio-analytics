import numpy as np

def stress_test(weights, shocks):
    """
    shocks: lista com variações percentuais (ex: -0,3, -0,2, etc)
    """
    return (weights * shocks).sum()
"""
This script stores the helper functions that aid the main file.

Author: Harsh Parikh
"""

import numpy as np
from scipy.stats import norm


def black_scholes_analytical(spot, strike, sigma, r, T, option_type):
    """
    This function calculates the analytical solution for the Black-Scholes
    price for a vanilla European option.
    """
    # Calculating d1
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Calculating d2
    d2 = d1 - sigma * np.sqrt(T)

    # The value of sign would be `1` if call else `-1`for put.
    sign = 1 if option_type == "call" else -1

    return sign * spot * norm.cdf(sign * d1) - sign * np.exp(
        -r * T
    ) * strike * norm.cdf(sign * d2)

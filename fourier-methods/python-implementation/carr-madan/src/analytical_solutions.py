"""
Analytical Pricing Formulas for Stochastic Models.

This script provides closed-form solutions for various stochastic models 
(e.g., Black-Scholes) used to benchmark and validate the results of 
the Carr–Madan method.

Author: Harsh Parikh  
Date: July 27, 2025
"""

import warnings
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def bsm_option_price(spot, strike, sigma, r, T, t=0, option_type="call"):
    """
    Parameters:
    -----------
    spot: Initial stock price
    strike: Strike Price
    T: Time to maturity
    r: Risk Free interest rate
    sigma: Volatility
    """
    # Initial check
    if option_type not in ("put", "call"):
        raise ValueError("Please select one from `put` or `call`")

    # Calculating the time difference
    tau = T - t
    # Calculating d1 term
    d1 = (np.log(spot / strike) + (r + pow(sigma, 2) / 2) * tau) / (
        sigma * np.sqrt(tau)
    )

    # Calculating d2 term
    d2 = d1 - sigma * np.sqrt((tau))

    if option_type == "call":
        return spot * stats.norm.cdf(d1) - strike * np.exp(-r * tau) * stats.norm.cdf(
            d2
        )
    else:
        return strike * np.exp(-r * tau) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(
            -d1
        )

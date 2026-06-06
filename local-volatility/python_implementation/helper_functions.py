"""
This script acts as the helper function for storing the functions used 
to calculate the local volatility for the underlying ticker. 

Author: Harsh Parikh
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def calculate_bsm_price(spot, strike, sigma, r, T, option_type="call"):
    """
    This function calculates the analytical value of the Black-Scholes
    option contract.
    """
    # Calculating the value of d1
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / sigma * np.sqrt(T)

    # Calculating the value of d2
    d2 = d1 - np.sqrt(T) * sigma

    # The value of sign depends on the contract type. `1` for call and
    # `-1` for put

    sign = 1 if option_type == "call" else -1

    value = sign * spot * norm.cdf(sign * d1) - sign * strike * np.exp(
        -r * T
    ) * norm.cdf(sign * d2)

    return value


def bs_vega(spot, strike, T, r, q, sigma):
    """
    This function calculates the vega of the underlying option contract
    based on the Black-Scholes' analytical formula
    """
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return spot * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def calculate_implied_volatility(
    spot_price,
    strike_price,
    risk_free_rate,
    time_to_maturity,
    market_price,
    isCall=True,
):
    """
    Calculate the implied volatility using the Brent-Dekker method.
    """

    low, high = 1e-4, 5.0  # Bounds

    def objective_function(implied_vol):

        model_price = calculate_bsm_price(
            spot_price,
            strike_price,
            implied_vol,
            risk_free_rate,
            time_to_maturity,
            isCall,
        )

        return model_price - market_price

    try:
        return brentq(objective_function, a=low, b=high)
    except:
        raise ValueError("Unable to find implied volatility. Check input parameters.")


def calculate_iv(prices, strikes, ttms, spot, r, isCall=True):
    """
    This function is used to calculate the IVs based on the
    grid of strikes and maturities.

    Input params
    ------------
    prices: The array of prices in the form of i x j, where
            i is the index of maturity and j is the index of
            strike.
    strikes: Array of strike prices
    ttms: Array of maturities
    """
    IV = np.zeros_like(prices)
    for i, mat in enumerate(ttms):
        for j, strike in enumerate(strikes):
            IV[i, j] = calculate_implied_volatility(
                spot, strike, r, mat, prices[i, j], isCall=isCall
            )
    return IV

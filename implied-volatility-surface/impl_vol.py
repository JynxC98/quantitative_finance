"""
Script to calculate implied volatility
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def calculate_option_price(
    spot_price,
    strike_price,
    time_to_maturity,
    risk_free_rate,
    volatility,
    option_type="call",
):
    """
    Calculate the price of a European option using the Black-Scholes-Merton model.

    Parameters:
    -----------
    initial_price : float
        The current price of the underlying asset.
    strike_price : float
        The strike price of the option.
    time_to_maturity : float
        Time to maturity in years.
    risk_free_rate : float
        The risk-free interest rate (annualized).
    volatility : float
        The volatility of the underlying asset (annualized).
    option_type : str, optional
        The type of option, either "call" or "put" (default is "call").

    Returns:
    --------
    float
        The calculated option price.
    """

    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    d1 = (
        np.log(spot_price / strike_price)
        + (risk_free_rate + volatility**2 / 2) * time_to_maturity
    ) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    if option_type == "call":
        return spot_price * norm.cdf(d1) - strike_price * np.exp(
            -risk_free_rate * time_to_maturity
        ) * norm.cdf(d2)
    else:
        return strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(
            -d2
        ) - spot_price * norm.cdf(-d1)


def calculate_implied_volatility(
    spot_price,
    strike_price,
    risk_free_rate,
    time_to_maturity,
    market_price,
    option_type="call",
):
    """
    Calculate the implied volatility using the Brent-Dekker method.

    Parameters:
    -----------
    spot_price : float
        The current price of the underlying asset.
    strike_price : float
        The strike price of the option.
    risk_free_rate : float
        The risk-free interest rate (annualized).
    time_to_maturity : float
        Time to maturity in years.
    market_price : float
        The observed market price of the option.

    Returns:
    --------
    float or None
        The calculated implied volatility, or None if the calculation fails.
    """

    low, high = 1e-4, 5.0  # Bounds

    def objective_function(implied_vol):
        return (
            calculate_option_price(
                spot_price,
                strike_price,
                time_to_maturity,
                risk_free_rate,
                implied_vol,
                option_type,
            )
            - market_price
        )

    try:
        return brentq(objective_function, a=low, b=high)
    except:
        raise ValueError("Unable to find implied volatility. Check input parameters.")

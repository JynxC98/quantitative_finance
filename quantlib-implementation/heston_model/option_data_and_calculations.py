"""
Codes related to fetching option data and other calculations 
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf


def get_strike_price_pivot_table(
    ticker,
    maturity_min=0.1,
    maturity_max=2,
    moneyness_min=0.95,
    moneyness_max=1.2,
    option_type="call",
):
    """
    Generate a pivot table of option strike prices for a given ticker.

    This function fetches option data for a specified stock ticker and creates a pivot table
    of strike prices across different maturities. It filters the data based on time to maturity
    and moneyness (strike price relative to spot price).

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
        maturity_min (float, optional): Minimum time to maturity in years. Defaults to 0.1.
        maturity_max (float, optional): Maximum time to maturity in years. Defaults to 2.
        moneyness_min (float, optional): Minimum moneyness (strike/spot) to consider. Defaults to 0.95.
        moneyness_max (float, optional): Maximum moneyness (strike/spot) to consider. Defaults to 1.2.
        option_type (str, optional): The type of option ('call' or 'put'). Defaults to 'call'.

    Returns:
        Dictionary
    """
    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    option_data = yf.Ticker(ticker)
    spot_price = option_data.history("1D")["Close"].iloc[0]

    today = pd.Timestamp.today()
    valid_maturities = [
        mat
        for mat in option_data.options
        if maturity_min < (pd.to_datetime(mat) - today).days / 365 < maturity_max
    ]

    strikes_freq = defaultdict(int)
    all_data = []

    for maturity in valid_maturities:
        if option_type == "call":
            chain = option_data.option_chain(maturity).calls
        elif option_type == "put":
            chain = option_data.option_chain(maturity).puts
        ttm = (pd.to_datetime(maturity) - today).days / 365

        valid_strikes = chain[
            (chain["strike"] >= moneyness_min * spot_price)
            & (chain["strike"] <= moneyness_max * spot_price)
        ]

        for strike in valid_strikes["strike"]:
            strikes_freq[strike] += 1

        valid_strikes["TTM"] = ttm
        all_data.append(valid_strikes[["strike", "lastPrice", "TTM"]])

    common_strikes = {
        strike for strike, freq in strikes_freq.items() if freq == len(valid_maturities)
    }

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data[combined_data["strike"].isin(common_strikes)]

    pivot_table = combined_data.pivot_table(
        index="TTM", columns="strike", values="lastPrice", fill_value=0
    )

    data = {
        "Pivot Table": pivot_table,
        "Valid Maturities": valid_maturities,
        "spot_price": spot_price,
    }

    return data


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


def bsm_implied_volatility(spot, strike, risk_free, tau, market_price):
    """
    Calculate the implied volatility for a given option price using the BSM model.

    Parameters:
    -----------
    spot : float
        The current price of the underlying asset.
    strike : float
        The strike price of the option.
    risk_free : float
        The risk-free interest rate.
    market_price : float
        The market price of ith strike at jth maturity

    Returns:
    --------
    float
        The implied volatility that equates the Black-Scholes price to the Market price.

    Raises:
    -------
    ValueError
        If the root-finding algorithm fails to converge within the specified bounds.
    """
    low, high = 1e-4, 5.0

    def objective_function(imp_vol):
        """
        Define the objective function for root-finding.

        This function calculates the difference between the Black-Scholes price
        and the Heston model price for a given implied volatility.

        Parameters:
        -----------
        imp_vol : float
            The implied volatility to test.

        Returns:
        --------
        float
            The difference between Black-Scholes and Heston prices.
        """
        return (
            calculate_option_price(spot, strike, tau, risk_free, imp_vol) - market_price
        )

    return brentq(lambda x: objective_function(x), a=low, b=high)

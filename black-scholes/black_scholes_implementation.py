"""
Implementation of Black-Scholes model
"""
import warnings
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def call_option_price(initial_price, strike_price, T, rf, sigma):
    """
    Parameters:
    -----------
    S: Initial stock price
    E: Strike Price
    T: Time to maturity
    rf: Risk Free interest rate
    sigma: Volatility
    """
    d1 = (np.log(initial_price / strike_price) + (rf + pow(sigma, 2) / 2) * T) / (
        sigma * np.sqrt(T)
    )

    d2 = d1 - sigma * np.sqrt((T))

    return initial_price * stats.norm.cdf(d1) - strike_price * np.exp(
        -rf * T
    ) * stats.norm.cdf(d2)


def put_option_price(initial_price, strike_price, T, rf, sigma):
    """
    Parameters:
    -----------
    initial_price: Initial stock price
    strike_price: Strike Price
    T: Time to maturity
    rf: Risk Free interest rate
    sigma: Volatility
    """
    d1 = (np.log(initial_price / strike_price) + (rf + pow(sigma, 2) / 2) * T) / (
        sigma * np.sqrt(T)
    )

    d2 = d1 - sigma * np.sqrt((T))

    return -initial_price * stats.norm.cdf(-d1) + strike_price * np.exp(
        -rf * T
    ) * stats.norm.cdf(-d2)


if __name__ == "__main__":
    print(call_option_price(100, 100, 1, 0.05, 0.25))
    print(put_option_price(100, 100, 1, 0.05, 0.2))

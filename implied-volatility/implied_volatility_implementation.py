"""
A script to calculate implied volatility.
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


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

    return initial_price * norm.cdf(d1) - strike_price * np.exp(-rf * T) * norm.cdf(d2)


def implied_volatility(S, K, r, T, actual_price, callput):
    """
    Calculate implied volatility using Newton's iterative formula.

    Parameters
    ----------
    S : float
        Stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    actual_price : float
        Option price.
    callput : int
        Type of option. 1 for Call, 0 for Put.

    Returns
    -------
    float
        Implied volatility.
    """
    imp_vol = 1  # Initial guess
    max_iterations = 1000
    tolerance = 1e-5
    option_price = 0

    for _ in range(max_iterations):
        sigma = imp_vol
        d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if callput == 1:  # Call option
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif callput == 0:  # Put option
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        f_sigma = option_price - actual_price  # To calculate implied volatility
        vega = S * np.sqrt(T) * norm.pdf(d1)
        imp_vol = sigma - f_sigma / vega
        if abs(sigma - imp_vol) <= tolerance:
            break

    return imp_vol


if __name__ == "__main__":
    S = 100
    RISK_FREE = 0.05
    K = np.linspace(
        80, 120, 20
    )  # We create an array of strike prices to calculate actual call option prices.
    SIGMA = 0.25
    implied_volatility_values = np.zeros([len(K)])
    for itr, strike_price in enumerate(K):
        option_price = call_option_price(S, strike_price, 1, RISK_FREE, SIGMA)
        implied_volatility_values[itr] = implied_volatility(
            S, strike_price, RISK_FREE, 1, option_price, 1
        )

    plt.plot(
        K, implied_volatility_values, marker="o"
    )  # Added marker for better visualization
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Volatility Smile")
    plt.grid(True)
    plt.show()

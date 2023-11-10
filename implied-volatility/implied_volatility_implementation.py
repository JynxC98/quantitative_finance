"""
A script to calculate implied volatility.
"""
import numpy as np
from scipy.stats import norm


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
    K = np.linspace(80, 120, 20)
    # implied_volatility_values = np.zeros([len(K)])
    # for itr, strike_price in enumerate(K):
    #     implied_volatility_values[itr] = implied_volatility(S, strike_price, RISK_FREE, )

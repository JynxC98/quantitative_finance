"""
A script to perform Monte-Carlo simulations.

Author: Harsh Parikh
Date: 31-01-2025
"""

import numpy as np


def control_variate(spot, strike, sigma, r, T, M=5000, N=5000, isCall=True):
    """
    This function simulates Monte-Carlo paths to calculate the option payoff. A
    control variate is incorporated to reduce the variance of the option payoff.

    Input parameters
    ----------------
    spot: The current spot price.
    strike: The pre-determined strike price.
    sigma: Volatility
    r: Risk-free rate
    T: Time to maturity
    M: Number of Brownian paths
    N: Number of time steps
    """

    # This LOC ensures `np.float` is not displayed
    np.set_printoptions(legacy="1.25")

    # Calculating the time step
    dt = T / N

    # Initialising the option grid
    grid = np.ones((M, N + 1))

    # Calculating the drift
    drift = (r - 0.5 * sigma**2) * dt

    # Calculating the diffusion
    diffusion = sigma * np.sqrt(dt) * np.random.standard_normal(size=(M, N))

    # Initialising the inital value of the spot grid
    grid[:, 0] = spot

    # Generating the vectorised Brownian paths
    grid[:, 1:] = spot * np.cumprod(np.exp(drift + diffusion), axis=1)

    # Calculating the option payoff
    payoff = (
        np.maximum(grid[:, -1] - strike, 0.0)
        if isCall
        else np.maximum(strike - grid[:, -1])
    ) * np.exp(-r * T)

    # Calculating the statistics
    mean_price = np.mean(payoff)

    std_dev = np.std(
        payoff, ddof=1
    )  # ddof = 1 calculates the sample standard deviation

    # Assuming 95% confidence interval
    upper_limit = mean_price + 1.96 * std_dev / np.sqrt(M)
    lower_limit = mean_price - 1.96 * std_dev / np.sqrt(M)

    # Returning the statistics
    return {
        "mean option price": mean_price,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "standard deviation": std_dev,
    }


if __name__ == "__main__":
    SPOT = 100
    STRIKE = 120
    SIGMA = 0.3
    RATE = 0.045
    MAT = 1
    ISCALL = True

    option_statistics = control_variate(
        spot=SPOT, strike=STRIKE, sigma=SIGMA, r=RATE, T=MAT, isCall=ISCALL
    )

    print(option_statistics)

"""
A script to simulate arithmetic Asian options using Monte-Carlo simulations

Author: Harsh Parikh
Date: 03rd March 2025
"""

import numpy as np


def arithmetic_asian_option(
    spot, strike, sigma, T, r, M=5000, N=5000, option_type="call"
):
    """
    Prices an arithmetic Asian option using Monte Carlo simulation under the
    Geometric Brownian Motion (GBM) model.

    Parameters:
    ----------
    spot : float
        Initial spot price of the underlying asset.
    strike : float
        Strike price of the option.
    sigma : float
        Volatility of the underlying asset (annualized).
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (annualized).
    M : int, optional (default=5000)
        Number of Monte Carlo simulation paths.
    N : int, optional (default=5000)
        Number of time steps per path.
    option_type : str, optional (default="call")
        Type of option: "call" or "put".

    Returns:
    -------
    float
        Estimated price of the arithmetic Asian option.

    Notes:
    -----
    The payoff is based on the arithmetic average of the asset prices over time.
    The asset follows a Geometric Brownian Motion (GBM) process.
    """

    # Basic data input check

    if option_type not in ("call", "put"):
        raise ValueError("Please select an option from `call` or `put.")

    # Calculating the time discritisation
    dt = T / N

    # Initialising the spot grid
    spot_grid = np.zeros((M, N + 1))  # N + 1 as N = 0 is the spot price itself

    # Calculating the drift term
    drift = (r - 0.5 * sigma**2) * dt

    # Calculating the diffusion term
    diffusion = sigma * np.random.standard_normal(size=(M, N)) * np.sqrt(dt)

    # Initializing the first time step as spot price
    spot_grid[:, 0] = spot

    # Populating the grid using efficient numpy vectorisation
    spot_grid[:, 1:] = spot * np.cumprod(np.exp(drift + diffusion), axis=1)

    # Calculating the average stock price over the grid
    average_price = np.mean(spot_grid, axis=1)

    # Calculating the option payoff
    payoff = (
        np.maximum(average_price - strike, 0)
        if option_type == "call"
        else np.maximum(strike - average_price, 0)
    )

    # Calculating the mean discounted payoff
    required_payoff = np.exp(-r * T) * np.mean(payoff)

    return required_payoff


if __name__ == "__main__":
    SPOT = 100
    STRIKE = 110
    T = 1
    RISK_FREE = 0.045
    SIGMA = 0.25

    # Calculating the option price
    option_price = arithmetic_asian_option(
        spot=SPOT, strike=STRIKE, sigma=SIGMA, T=T, r=RISK_FREE
    )

    print(option_price)

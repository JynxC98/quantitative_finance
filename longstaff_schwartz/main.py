"""
An implementation of the Longstaff Schwartz Model (LSM) to price American options
efficiently. The LSM model uses dynamic programming to find the optimal stopping 
point and uses Monte-Carlo simulations to calculate the expected value of the option.

References:
1. https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf
2. https://uu.diva-portal.org/smash/get/diva2:818128/FULLTEXT01.pdf

Date: 13-December-2024

Author: Harsh Parikh
"""

import numpy as np
from statsmodels.regression.linear_model import OLS


def basis_function(k, X):
    """
    The basis function is to simulate the Laguerre polynomials.

    Input
    -----
    X (np.array): Input vector
    k (int): The number of Laguerre polynomials

    Returns
    -------
    tuple:
        (1, L_k)
    """
    if type(X) is not np.ndarray:
        raise TypeError("The type of the vector X should be np.ndarray")

    if k > 4:
        raise ValueError("The value of k should be less than equal to 4")

    if k == 1:
        return tuple([np.ones(X.size), (1 - X)])

    elif k == 2:
        laguerre_poly = [np.ones(X.size), (1 - X), 1 / 2 * (2 - 4 * X + X**2)]
        return tuple(laguerre_poly)

    elif k == 3:
        laguerre_poly = [
            np.ones(X.size),
            (1 - X),
            1 / 2 * (2 - 4 * X + X**2),
            1 / 6 * (6 - 18 * X + 9 * X**2 - X**3),
        ]
        return tuple(laguerre_poly)

    elif k == 4:
        laguerre_poly = [
            np.ones(X.size),
            (1 - X),
            1 / 2 * (2 - 4 * X + X**2),
            1 / 6 * (6 - 18 * X + 9 * X**2 - X**3),
            1 / 24 * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4),
        ]
        return tuple(laguerre_poly)


def longstaff_schwartz(spot, strike, sigma, T, r, N=500, M=500, isCall=True):
    r"""
    The function to implement the Longstaff Schwartz pricing engine. The engine
    assumes that the underlying follows a Geometric Brownian Motion (GBM) with constant values
    of $\mu$ and $\sigma$.

    The dynamics of the GBM are given as:

    $$dS_t = S_t (\mu \, dt + \sigma \, dW_t)$$

    where:
    - $\mu$: Drift
    - $\sigma$: Volatility
    - $dW$: Wiener process.


    Input Parameters
    ----------------
    spot: The current spot price\
    strike: The predetermined strike price\
    sigma: The current volatility\
    T: Time to maturity\
    r: Risk-free rate\
    N (optional): Number of time steps\
    M (optional): Number of Monte-Carlo paths\
    isCall (optional): `True` for call option and `False` for put

    Returns
    -------
    float:
        The price of the corresponsing American option.
    """

    # Discretising the time grid
    dt = T / N

    # This variable stores M paths and N time steps. Here, `+1` is to account
    # the first value for the spot simulation.

    stock_paths = np.zeros((M + 1, N))

    # Initializing the first value of the grid as the spot price at time T = 0

    stock_paths[0:,] = spot

    # Calculating the drift and diffusion terms for the geometric Brownian motion.

    drift = (r - 0.5 * sigma**2) * dt

    dZ = np.random.normal(0, 1.0, size=(M, N)) * np.sqrt(
        dt
    )  # Standard normal variable.

    diffusion = sigma * dZ

    # Generating the stock paths
    stock_paths[1:,] = spot * np.exp(np.cumsum((drift + diffusion), axis=0))

    # Calculating the option prices
    option_prices = (
        np.max(stock_paths - strike, 0.0)
        if isCall
        else np.max(strike - option_prices, 0.0)
    )

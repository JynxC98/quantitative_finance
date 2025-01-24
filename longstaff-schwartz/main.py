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
import statsmodels.api as sm


def basis_function(k, X):
    """
    Generate Laguerre polynomial basis functions.

    Input
    -----
    X (np.array): Input vector
    k (int): The number of Laguerre polynomials

    Returns
    -------
    tuple:
        Tuple of basis function arrays
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("The type of the vector X should be np.ndarray")

    if k > 4:
        raise ValueError("The value of k should be less than or equal to 4")

    # Laguerre polynomial basis functions
    basis_funcs = [
        [np.ones(X.size), (1 - X)],
        [np.ones(X.size), (1 - X), 1 / 2 * (2 - 4 * X + X**2)],
        [
            np.ones(X.size),
            (1 - X),
            1 / 2 * (2 - 4 * X + X**2),
            1 / 6 * (6 - 18 * X + 9 * X**2 - X**3),
        ],
        [
            np.ones(X.size),
            (1 - X),
            1 / 2 * (2 - 4 * X + X**2),
            1 / 6 * (6 - 18 * X + 9 * X**2 - X**3),
            1 / 24 * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4),
        ],
    ]

    return tuple(basis_funcs[k - 1])


def longstaff_schwartz(
    spot, strike, sigma, T, r, N=1000, M=10000, isCall=True, k=3, seed=42
):
    r"""
    Implement the Longstaff Schwartz pricing engine for American options.

    The engine assumes the underlying follows a Geometric Brownian Motion (GBM)
    with constant values of $\mu$ and $\sigma$.

    The dynamics of the GBM are given as:

    $$dS_t = S_t (\mu \, dt + \sigma \, dW_t)$$

    where:
    - $\mu$: Drift
    - $\sigma$: Volatility
    - $dW$: Wiener process.

    Input Parameters
    ----------------
    spot : float
        The current spot price
    strike : float
        The predetermined strike price
    sigma : float
        The current volatility
    T : float
        Time to maturity
    r : float
        Risk-free rate
    N : int, optional
        Number of time steps (default 500)
    M : int, optional
        Number of Monte-Carlo paths (default 10000)
    isCall : bool, optional
        `True` for call option and `False` for put (default True)
    k : int, optional
        Number of Laguerre polynomials (default 3)
    seed : int, optional
        Random seed for reproducibility (default 42)

    Returns
    -------
    float:
        The price of the corresponding American option.
    """
    # Initialising a random seed for reproducability.
    np.random.seed(seed)

    # Discretising the time grid
    dt = T / N

    # Initialising the stock paths.
    stock_paths = np.zeros((M, N + 1))  # M paths, N+1 time steps

    stock_paths[:, 0] = spot  # Initial spot price at t=0

    # Initialising the drift and diffusion terms
    drift = (r - 0.5 * sigma**2) * dt

    dZ = np.random.normal(0, 1.0, size=(M, N)) * np.sqrt(dt)  # Brownian increments

    diffusion = sigma * dZ

    # Generating vectorised form of stock paths.
    stock_paths[:, 1:] = spot * np.cumprod(np.exp(drift + diffusion), axis=1)

    # Calculating the option payoff based on the option type.
    option_prices = (
        np.maximum(stock_paths - strike, 0.0)
        if isCall
        else np.maximum(strike - stock_paths, 0.0)
    )

    # Initialising the cash flows with the terminal payoff
    cash_flows = option_prices[:, -1]

    # Interating backwards

    for t in range(N - 1, 0, -1):

        # Storing the in-the-money paths
        in_the_money = option_prices[:, t] > 0

        if not np.any(in_the_money):
            # Discounting the cash flows part
            cash_flows *= np.exp(-r * dt)
            continue

        # Current stock prices for in-the-money paths
        X = stock_paths[in_the_money, t]

        # Discounted future cash flows
        Y = cash_flows[in_the_money] * np.exp(-r * dt)

        # Calculating the basis function
        basis = np.column_stack(basis_function(k, X / spot))

        # Least squares regression to estimate continuation value using OLS
        model = sm.OLS(Y, basis).fit()

        # Get the continuation values
        continuation_value = model.predict(basis)

        # Intrinsic value at current time step
        intrinsic_value = option_prices[in_the_money, t]

        # Determine optimal exercise
        exercise = intrinsic_value > continuation_value

        # Update cash flows
        cash_flows[in_the_money] = np.where(
            exercise, intrinsic_value, cash_flows[in_the_money] * np.exp(-r * dt)
        )

    # Final option price calculation
    option_price = np.mean(cash_flows) * np.exp(-r * dt)
    return option_price


if __name__ == "__main__":
    # Underlying parameters
    SPOT = 100
    STRIKE = 110
    RISK_FREE = 0.045
    SIGMA = 0.25
    T = 1

    # Longstaff-Schwartz American Option Price
    lsm_option_value = longstaff_schwartz(
        spot=SPOT, strike=STRIKE, sigma=SIGMA, T=T, r=RISK_FREE
    )
    print(lsm_option_value)

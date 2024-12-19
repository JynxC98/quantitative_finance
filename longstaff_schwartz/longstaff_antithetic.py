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


def longstaff_schwartz(spot, strike, sigma, T, r, N=1000, M=10000, isCall=True, k=3):
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

    # Discretising the time grid
    dt = T / N

    stock_paths = np.zeros((M, N + 1))

    stock_paths[:, 0] = spot

    # Calculating the drift term
    drift = (r - 0.5 * sigma**2) * dt

    # Calculating the diffusion term for normal and antithetic variable
    dZ = np.random.normal(0, 1.0, size=(M // 2, N)) * np.sqrt(dt)
    diffusion = sigma * dZ
    antithetic_diffusion = sigma * (-dZ)

    # Generating stock paths

    # Generating stock paths with cumulative product
    stock_paths[: M // 2, 1:] = spot * np.cumprod(
        np.exp(drift + diffusion), axis=1
    )  # Normal path

    stock_paths[M // 2 :, 1:] = spot * np.cumprod(
        np.exp(drift + antithetic_diffusion), axis=1
    )  # Antithetic path

    # Calculating option payoffs
    option_prices = (
        np.maximum(stock_paths - strike, 0.0)
        if isCall
        else np.maximum(strike - stock_paths, 0.0)
    )

    # Initialising cash flows with terminal payoff
    cash_flows = option_prices[:, -1]

    # Iterating backwards
    for t in range(N - 1, 0, -1):
        in_the_money = option_prices[:, t] > 0
        if not np.any(in_the_money):

            # Discounting the previous cashflows
            # cash_flows = cash_flows * np.exp(-r * dt)
            continue

        X = stock_paths[in_the_money, t]
        Y = cash_flows[in_the_money] * np.exp(-r * dt)

        # Basis functions and regression
        basis = np.column_stack(basis_function(k, X / spot))
        model = sm.OLS(Y, basis).fit()

        # Continuation and intrinsic values
        continuation_value = model.predict(basis)
        intrinsic_value = option_prices[in_the_money, t]

        # Update cash flows
        exercise = intrinsic_value > continuation_value
        cash_flows[in_the_money] = np.where(
            exercise, intrinsic_value, cash_flows[in_the_money] * np.exp(-r * dt)
        )

    # Calculate the option price as the mean of all cash flows
    option_price = np.mean(cash_flows) * np.exp(-r * dt)

    return option_price


if __name__ == "__main__":
    # Underlying parameters
    SPOT = 100
    STRIKE = 110
    RISK_FREE = 0.045
    SIGMA = 0.25
    T = 1

    # Longstaff-Schwartz American Option Price with Antithetic Variance Reduction
    lsm_option_value = longstaff_schwartz(
        spot=SPOT, strike=STRIKE, sigma=SIGMA, T=T, r=RISK_FREE
    )
    print("Option Price with Antithetic Variance Reduction:", lsm_option_value)

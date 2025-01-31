"""
A script to store all the methods of the option pricing.

Author: Harsh Parikh
Date: 31-01-2024
"""

import numpy as np
import scipy.stats as stats


def calculate_option_price_bsm(spot, strike, sigma, rf, T, isCall=True):
    """
    Parameters:
    -----------
    S: Initial stock price
    E: Strike Price
    T: Time to maturity
    rf: Risk Free interest rate
    sigma: Volatility
    """

    d1 = (np.log(spot / strike) + (rf + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt((T))

    if isCall:
        return spot * stats.norm.cdf(d1) - strike * np.exp(-rf * T) * stats.norm.cdf(d2)
    else:
        return strike * np.exp(-rf * T) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(
            d1
        )


def antithetic_method(spot, strike, sigma, r, T, M=5000, N=5000, isCall=True):
    """
    This function simulates Monte-Carlo paths to calculate the option payoff. An
    antithetic variable is incorporated to reduce the variance of the option payoff.

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

    # Generating the Brownian increments
    dZ = np.random.standard_normal(size=(M // 2, N))

    dZ_antithetic = -dZ
    # Calculating the diffusion terms
    diffusion = sigma * np.sqrt(dt) * dZ
    diffusion_antithetic = sigma * np.sqrt(dt) * dZ_antithetic

    # Initialising the inital value of the spot grid
    grid[:, 0] = spot

    # Generating the vectorised Brownian paths
    grid[: M // 2, 1:] = spot * np.cumprod(np.exp(drift + diffusion), axis=1)
    grid[M // 2 :, 1:] = spot * np.cumprod(np.exp(drift + diffusion_antithetic), axis=1)

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
        "mean_price": mean_price,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "std_dev": std_dev,
    }


def monte_carlo_sim(spot, strike, sigma, r, T, M=5000, N=5000, isCall=True):
    """
    This function simulates Monte-Carlo paths to calculate the option payoff.

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

    std_dev = np.std(payoff, ddof=1)  # ddof = 1 calculates the sample std_dev

    # Assuming 95% confidence interval
    upper_limit = mean_price + 1.96 * std_dev / np.sqrt(M)
    lower_limit = mean_price - 1.96 * std_dev / np.sqrt(M)

    # Returning the statistics
    return {
        "mean_price": mean_price,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "std_dev": std_dev,
    }


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
    )

    # Using the equivalent martingale measure as the control variable
    control_variable = spot * np.exp(r * T) * np.ones(M)

    # Calculating the control variate
    difference = grid[:, -1] - control_variable

    # Calculating the control variate coefficient
    beta = -np.cov(payoff, grid[:, -1])[0, 1] / np.var(grid[:, -1])

    # Updating the payoff
    updated_payoff = np.exp(-r * T) * (payoff + beta * (difference))

    # Calculating the statistics
    mean_price = np.mean(updated_payoff)

    std_dev = np.std(updated_payoff, ddof=1)  # ddof = 1 calculates the sample std_dev

    # Assuming 95% confidence interval
    upper_limit = mean_price + 1.96 * std_dev / np.sqrt(M)
    lower_limit = mean_price - 1.96 * std_dev / np.sqrt(M)

    # Returning the statistics
    return {
        "mean_price": mean_price,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "std_dev": std_dev,
        "coefficient": beta,
    }


if __name__ == "__main__":
    SPOT = 100
    STRIKE = 120
    SIGMA = 0.3
    RATE = 0.045
    MAT = 1
    ISCALL = True

    option_statistics = monte_carlo_sim(
        spot=SPOT, strike=STRIKE, sigma=SIGMA, r=RATE, T=MAT, isCall=ISCALL
    )

    print(option_statistics)

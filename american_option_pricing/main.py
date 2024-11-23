"""
Binomial method to calculate the American option using Bellman's equation.
"""

import numpy as np


def american_binomial(spot, strike, rate, T, sigma, num_steps, option_type="call"):
    """
    Calculates the value of an American option using the binomial tree method and Bellman's equation.
    """

    if option_type not in ("call", "put"):
        raise ValueError("Please select `call` or `put`")

    # Evaluating the timestep

    dt = T / num_steps

    # Calculating the up factor using CRR parametrisation

    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Calculating the risk neurtal probability

    p = (np.exp(rate * dt) - d) / (u - d)
    # Storing all possible values of the stock price in the given time framework.
    spot_grid = np.zeros(num_steps)

    for itr in range(num_steps + 1):

        spot_grid[itr] = spot * (u ** (num_steps - itr)) * (d**itr)

    # Precalculating the option payoffs for an efficient calculation

    if option_type == "call":
        payoffs = np.max(spot_grid - strike, 0)
    else:
        payoffs = np.max(strike - spot_grid, 0)

    # Backward calculation for options

    for itr in range(num_steps - 1, -1, -1):
        pass

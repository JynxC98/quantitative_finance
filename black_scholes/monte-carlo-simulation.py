"""Simulation of monte carlo simulation
"""

import collections
import warnings

warnings.filterwarnings("ignore")

# External Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 100


def stock_monete_carlo(
    S_0: float, mu: float, sigma: float, n_iter: int = 1000
) -> pd.DataFrame:
    """
    Implementation of monte-carlo simulation

    Parameters
    ----------
    mu: Mean of the stock
    sigma: Standard deviation of the stock
    n_iter: number of iterations
    S_0: Initial price of the stock.

    Output
    ------
    Graph of several simulations and the mean of the simulations
    """

    result = []

    for _ in range(NUM_SIMULATIONS):
        prices = [S_0]

        for _ in range(n_iter):
            stock_price = prices[-1] * np.exp(
                (mu - 0.5 * pow(sigma, 2)) + sigma * np.random.normal()
            )

            prices.append(stock_price)

        result.append(prices)

    simulation_data = pd.DataFrame(result).T

    return simulation_data


if __name__ == "__main__":
    print(stock_monete_carlo(300, 0.00034, 0.02))

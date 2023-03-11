import numpy as np
import pandas as pd


class OptionPricing:
    """
    Simulation of option pricing using Monte Carlo Simulations

    Parameters
    ----------
    S0: Initial price of the stock
    E : Strike price
    T : Time to maturity
    rf: risk free interest rate
    sigma: Volatility
    """

    np.random.seed(100)

    NUM_ITERATIONS = 10000

    def __init__(self, S0: float, T: int, E: float, rf: float, sigma: float) -> None:
        """
        Initialisation class of the option pricing simulation.
        """
        self.S0 = S0
        self.T = T
        self.E = E
        self.rf = rf
        self.sigma = sigma

    def call_option_simulation(self):
        """
        Simuates the call option pricing.
        """

        option_data = np.zeros(
            [self.NUM_ITERATIONS, 2]
        )  # First columns stores the 0s and second column stores the payoff.

        random_number = np.random.normal(
            0, 1, [1, self.NUM_ITERATIONS]
        )  # 1 dimensional array with as many items as the iterations

        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * pow(self.sigma, 2))
            + self.sigma
            + np.sqrt(self.T) * random_number
        )
        option_data[:, 1] = stock_price - self.E

        average = np.sum(np.amax(option_data, axis=1)) / float(self.NUM_ITERATIONS)
        return option_data


if __name__ == "__main__":
    option = OptionPricing(100, 100, 1, 0.06, 0.2)
    print(option.call_option_simulation())

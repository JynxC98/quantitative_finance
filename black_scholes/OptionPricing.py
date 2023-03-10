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

    NUM_ITERATIONS = 10000

    def __init__(self, S0: float, E: float, rf: float, sigma: float) -> None:
        """
        Initialisation class of the option pricing simulation.
        """
        self.S0 = S0
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
        return option_data

    if __name__ == "__main__":
        pass

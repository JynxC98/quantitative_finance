"""
Script for calculating the option price and implied volatility.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class Option:
    """
    Class representing an option for pricing and implied volatility calculations.
    """

    def __init__(self, spot, strike, risk_free, maturity, option_type=1):
        """
        Initializes the Option instance.

        Parameters:
        - spot: float, current price of the underlying asset
        - strike: float, strike price of the option
        - risk_free: float, risk-free interest rate
        - maturity: float, time to maturity (in years)
        - option_type: int, 1 for Call option, 0 for Put option
        """
        self.spot = spot
        self.strike = strike
        self.risk_free = risk_free
        self.maturity = maturity
        self.option_type = option_type

        if option_type not in (0, 1):
            raise ValueError("Invalid option_type, please select 0 (Put) or 1 (Call)")

    def calculate_price(self, sigma):
        """
        Calculates the price of the option based on the Black-Scholes pricing model.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, price of the option
        """
        d1 = (
            np.log(self.spot / self.strike)
            + (self.risk_free + 0.5 * sigma**2) * self.maturity
        ) / (sigma * np.sqrt(self.maturity))
        d2 = d1 - sigma * np.sqrt(self.maturity)

        if self.option_type == 1:
            return self.spot * norm.cdf(d1) - self.strike * np.exp(
                -self.risk_free * self.maturity
            ) * norm.cdf(d2)
        return self.strike * np.exp(-self.risk_free * self.maturity) * norm.cdf(
            -d2
        ) - self.spot * norm.cdf(-d1)

    def implied_volatility(self, actual_price):
        """
        Calculates the implied volatility of the option given the market price.

        Parameters:
        - actual_price: float, market price of the option

        Returns:
        - float, implied volatility of the option
        """
        if actual_price == 0:
            return 0.0

        def objective_function(sigma):
            """
            Objective function to calculate implied volatility.

            Parameters:
            - sigma: float, volatility of the option

            Returns:
            - float, difference between calculated and actual option price
            """
            return self.calculate_price(sigma) - actual_price

        low, high = 1e-4, 5.0
        try:
            return brentq(objective_function, low, high, xtol=1e-6)
        except ValueError as error:
            print(
                f"ValueError for strike={self.strike}, maturity={self.maturity}: {error}"
            )
        except RuntimeError as error:
            print(
                f"RuntimeError for strike={self.strike}, maturity={self.maturity}: {error}"
            )
        return np.nan

"""
Script for calculating the option price, implied volatility, and option Greeks.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class Option:
    """
    Class representing an option for pricing, implied volatility, and Greeks calculations.
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

    def _d1_d2(self, sigma):
        """
        Calculates the d1 and d2 parameters used in the Black-Scholes formula.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - d1: float, parameter d1
        - d2: float, parameter d2
        """
        d1 = (
            np.log(self.spot / self.strike)
            + (self.risk_free + 0.5 * sigma**2) * self.maturity
        ) / (sigma * np.sqrt(self.maturity))
        d2 = d1 - sigma * np.sqrt(self.maturity)
        return d1, d2

    def calculate_price(self, sigma):
        """
        Calculates the price of the option based on the Black-Scholes pricing model.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, price of the option
        """
        d1, d2 = self._d1_d2(sigma)

        if self.option_type == 1:  # Call option
            return self.spot * norm.cdf(d1) - self.strike * np.exp(
                -self.risk_free * self.maturity
            ) * norm.cdf(d2)
        else:  # Put option
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

    def delta(self, sigma):
        """
        Calculates the Delta of the option.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, Delta of the option
        """
        d1, _ = self._d1_d2(sigma)
        if self.option_type == 1:  # Call option
            return norm.cdf(d1)
        else:  # Put option
            return norm.cdf(d1) - 1

    def gamma(self, sigma):
        """
        Calculates the Gamma of the option.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, Gamma of the option
        """
        d1, _ = self._d1_d2(sigma)
        return norm.pdf(d1) / (self.spot * sigma * np.sqrt(self.maturity))

    def vega(self, sigma):
        """
        Calculates the Vega of the option.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, Vega of the option
        """
        d1, _ = self._d1_d2(sigma)
        return self.spot * norm.pdf(d1) * np.sqrt(self.maturity)

    def rho(self, sigma):
        """
        Calculates the Rho of the option.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, Rho of the option
        """
        _, d2 = self._d1_d2(sigma)
        if self.option_type == 1:  # Call option
            return (
                self.strike
                * self.maturity
                * np.exp(-self.risk_free * self.maturity)
                * norm.cdf(d2)
            )
        return (
            -self.strike
            * self.maturity
            * np.exp(-self.risk_free * self.maturity)
            * norm.cdf(-d2)
        )

    def theta(self, sigma):
        """
        Calculates the Theta of the option.

        Parameters:
        - sigma: float, volatility of the underlying asset

        Returns:
        - float, Theta of the option
        """
        d1, d2 = self._d1_d2(sigma)
        if self.option_type == 1:  # Call option
            theta = -self.spot * norm.pdf(d1) * sigma / (
                2 * np.sqrt(self.maturity)
            ) - self.risk_free * self.strike * np.exp(
                -self.risk_free * self.maturity
            ) * norm.cdf(
                d2
            )
        else:  # Put option
            theta = -self.spot * norm.pdf(d1) * sigma / (
                2 * np.sqrt(self.maturity)
            ) + self.risk_free * self.strike * np.exp(
                -self.risk_free * self.maturity
            ) * norm.cdf(
                -d2
            )
        return theta / 252  # Annualized theta converted to daily theta

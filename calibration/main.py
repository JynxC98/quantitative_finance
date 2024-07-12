"""
The main script for calibration.
"""

from scipy.optimize import minimize

from models import heston_call_price
from fetch_data import get_strike_price_pivot_table


class CalibrateModel:
    """
    Calibrates the Heston stochastic volatility model under the risk-neutral pricing measure.

    This class implements the calibration of the Heston model parameters to match observed
    market option prices. The Heston model is a popular stochastic volatility model used
    in quantitative finance for option pricing and risk management.

    The class performs the following main tasks:
    1. Calculates option prices using the Heston model under the risk-neutral measure.
    2. Calibrates the model parameters to minimise the difference between model and market prices.

    Key features:
    - Implements the Heston model pricing formula for European options.
    - Uses numerical optimisation techniques to find the best-fit model parameters.
    - Supports calibration across multiple strike prices and maturities simultaneously.
    - Provides tools for analysing the calibration results and assessing model fit.

    The Heston model parameters calibrated include:
    - v0: Initial variance
    - kappa: Mean reversion speed of variance
    - theta: Long-term mean of variance
    - sigma: Volatility of variance
    - rho: Correlation between stock price and variance processes
    - lambda: Price of volatility risk (optional, depending on implementation)

    Usage:
    This class is typically used in financial applications for:
    - Pricing exotic options
    - Calculating implied volatility surfaces
    - Risk management and scenario analysis
    - Studying the dynamics of asset price volatility

    Note:
    The calibration process can be computationally intensive and may require
    careful initialisation of parameters and choice of optimisation algorithm.
    """

    def __init__(self, ticker):
        """
        Initialisation module of `CalibrateModel` class.

        Input Parameters
        ----------------
            - ticker(str): stock ticker for a particular asset class.
        """
        self.ticker = ticker

        # Parameters to be calculated later
        self.parameter_history = []
        self.params_ = []


if __name__ == "__main__":
    TICKER = "AAPL"
    price, required_table = get_strike_price_pivot_table(ticker=TICKER)
    required_strikes = required_table.columns.values
    required_maturities = required_table.index.values
    market_prices = required_table.values
    # print(
    #     calibrate_model(
    #         spot_price=price,
    #         strikes=required_strikes,
    #         maturities=required_maturities,
    #         call_option_prices=market_prices,
    #     )
    # )

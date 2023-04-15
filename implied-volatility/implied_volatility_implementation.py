"""
Script to simulate implied volatility
"""
import warnings
import plotly.express as px
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")


def get_data_from_yahoo(start_date, end_date, ticker) -> np.array:
    """
    start_date: Historical start date of the data
    end_date: Historical end date of the data
    ticker: Ticker of the company needed to calculate the option data
    """
    data = yf.download(ticker, start_date, end_date)
    return np.log(1 + data["Adj Close"].pct_change())


class ImpliedVolatility:
    """
    Implied volatility calculation using BSM model

    Parameters
    ----------
    stock_price: Stock Price at time t
    risk_free_rate: Risk free interest rate
    strike_price: Strike price at time T
    time: Time to maturity
    sigma: Volatility at time t
    option_type: Call (1) or put(0) option.
    start: starting date
    end: ending date
    """

    OPTION_PRICE = 0

    def __init__(self, **kwargs) -> None:
        """
        Initialisation function of the class `Implied Volatility`.
        """
        if len(kwargs) != 6:
            raise ValueError("Incorrect number of arguments")

        self.start = kwargs.get("start")
        self.end = kwargs.get("end")
        self.stock_price = kwargs.get("stock_price")
        self.risk_free_rate = kwargs.get("risk_free_rate")
        self.strike_price = kwargs.get("strike_price")
        self.time = kwargs.get("time")
        self.sigma = kwargs.get("sigma")
        self.option_type = kwargs.get("option_type")

        if self.option_type not in (0, 1):
            raise ValueError("Incorrect input for option_type")

    def black_scholes_merton_price(self) -> float:
        """
        Calculates the value of option using BSM pricing formula.
        """
        d_1 = (
            np.log(self.stock_price / self.strike_price)
            + (self.risk_free_rate + 0.5 * pow(self.sigma, 2))
        ) / (self.sigma * np.sqrt(self.time))

        d_2 = d_1 - self.sigma * np.sqrt(self.time)

        if self.option_type == 1:
            return self.stock_price * norm.cdf(d_1) - self.strike_price * np.exp(
                -self.risk_free_rate * self.time
            ) * norm(d_2)
        else:
            return self.strike_price * np.exp(
                -self.risk_free_rate * self.time
            ) * norm.cdf(-d_2) - self.stock_price * norm.cdf(-d_1)

    def calculate_implied_volatility(self):
        """
        Uses Newton's iterative method to calculate implied volatility.
        """
        return brentq(
            lambda x: self.black_scholes_merton_price() - self.OPTION_PRICE, 0.00001, 2
        )

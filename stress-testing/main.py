"""
A script that generates optimal weights for the portfolio.
"""
from datetime import datetime, timedelta
import cvxpy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf


def generate_stress_scenarios(mean_returns, cov_matrix, num_scenarios=10000):
    """
    This function generates random cases of overall market fluctuations.
    """
    np.random.seed(42)
    return np.random.multivariate_normal(mean_returns, cov_matrix, num_scenarios)


class Portfolio:
    """
    A class for optimising a portfolio using data collected from Yahoo Finance.

    Attributes:
        - stocks (list): List of stock symbols for portfolio construction.
        - start_date (str): Start date for collecting historical data in 'YYYY-MM-DD' format.
        - end_date (str): End date for collecting historical data in 'YYYY-MM-DD' format.

    Methods:
        - collect_data(): Collects historical price data for the specified stocks from Yahoo Finance.
        - optimize_portfolio(): Minimizes portfolio volatility subject to specified returns and stress test.
        - generate_plots(): Generates appropriate plots illustrating the optimized portfolio.
    """

    def __init__(self, stocks, start_date, end_date):
        """
        Initialize the PortfolioOptimizer object.

        Parameters:
            - stocks (list): List of stock symbols for portfolio construction.
            - start_date (str): Start date for collecting historical data in 'YYYY-MM-DD' format.
            - end_date (str): End date for collecting historical data in 'YYYY-MM-DD' format.
        """
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def get_data_from_yahoo(self) -> pd.DataFrame:
        """
        Downloads data from yfinance.
        Parameters:
        start: Start Date (yyyy-mm-dd) format
        end: End Date (yyyy-mm-dd) format
        """
        stock_data = {}
        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(
                start=self.start_date, end=self.end_date
            )["Close"]

        return pd.DataFrame(stock_data)

    def calculate_returns(self) -> np.ndarray:
        """
        Calculates logarithmic returns of the historical data.
        """
        return_data = self.get_data_from_yahoo()
        return np.log(return_data / return_data.shift(1)).dropna()


if __name__ == "__main__":
    STOCKS = ["AAPL", "MSFT", "NVDA"]
    END = datetime.today()
    START = END - timedelta(10 * 365)  # data of 10 years
    portfolio = Portfolio(stocks=STOCKS, start_date=START, end_date=END)
    data = portfolio.calculate_returns()
    stress_scenarios = generate_stress_scenarios(
        np.mean(data.to_numpy(), axis=0), np.cov(data, rowvar=False)
    )
    print(stress_scenarios)

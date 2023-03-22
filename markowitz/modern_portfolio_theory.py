""" A script to generate optimal portfolio.
"""


import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from alive_progress import alive_bar
import scipy.optimize as optimization

warnings.filterwarnings("ignore")


class Portfolio:
    """
    Optimising portfolio to minimize risk.
    Input parameters:
    1. Stocks[Tickers]
    2. Start Date: The date from which the historical data is required.
    3. End Date: End point of the historical data.
    """

    NUM_TRADING_DAYS = 252
    NUM_PORTFOLIO = 100

    def __init__(self, stocks: list, start: str, end: str) -> None:
        """
        Initalisation class of the portfolio
        """
        self.stocks = stocks
        self.start = start
        self.end = end
        self.weights = None
        self.portfolio_data = None

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
            stock_data[stock] = ticker.history(start=self.start, end=self.end)["Close"]

        return pd.DataFrame(stock_data)

    def show_data(self) -> None:
        """
        Plots the line graph of price vs time for all the stocks in the portfolio.
        """
        data = self.get_data_from_yahoo()
        data.plot(figsize=(10, 5))
        plt.xlabel("Year")
        plt.ylabel("Price")
        plt.show()

    def calculate_return(self) -> float:
        """
        Returns the logarithmic price change of the price.
        """
        data = self.get_data_from_yahoo()
        log_return = np.log(data / data.shift(1))
        return log_return[1:]  # We skip the first row to eliminate the NaN values.

    def generate_portfolios(self) -> dict:
        """
        Generates random portfolios and their scatterplots.
        """
        portfolio_data = defaultdict(list)
        weights = []
        returns = self.calculate_return()

        for _ in range(self.NUM_PORTFOLIO):
            weight = np.random.random(len(self.stocks))
            weight /= np.sum(weight)
            weights.append(weight)

            portfolio_return = np.sum(returns.mean() * weight) * self.NUM_TRADING_DAYS
            portfolio_data["mean"].append(portfolio_return)
            portfolio_volatility = np.sqrt(
                np.dot(weight.T, np.dot(returns.cov() * self.NUM_TRADING_DAYS, weight))
            )
            portfolio_data["risk"].append(portfolio_volatility)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            portfolio_data["risk"],
            portfolio_data["mean"],
            c=np.array(portfolio_data["mean"]) / np.array(portfolio_data["risk"]),
            marker="o",
        )
        plt.grid(True)
        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Return")
        plt.colorbar(label="Sharpe Ratio")
        plt.show()

        self.weights = np.array(weights)
        self.portfolio_data = portfolio_data

        return portfolio_data

    def statistics(self) -> np.array:
        """
        Calculates the required statistics for optimisation function.
        """
        returns = self.calculate_return()
        portfolio_return = (
            np.sum(np.dot(returns.mean(), self.weights.T)) * self.NUM_TRADING_DAYS
        )
        portfolio_volatility = np.sqrt(
            np.dot(
                self.weights,
                np.dot(returns.cov() * self.NUM_TRADING_DAYS, self.weights.T),
            )
        )
        return np.array(
            [
                portfolio_return,
                portfolio_volatility,
                portfolio_return / portfolio_volatility,
            ]
        )

    def minimise_function(self) -> np.array:
        """
        Minimisation class for the given function
        """
        return -np.array(
            self.statistics()[-2]
        )  # The maximum of f(x) is the minimum of -f(x)

    def optimize_portfolio(self) -> np.array:
        """
        Used to optimize the weights with respect to the sharpe ratio.
        """
        data_portfolio = self.generate_portfolios()

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # The weights can at the most be 1.
        bounds = tuple((0, 1) for _ in range(len(self.stocks)))

        optimum = optimization.minimize(
            fun=self.minimise_function(),
            x0=np.array(self.weights),
            args=data_portfolio["mean"],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return optimum["x"].round(4)


if __name__ == "__main__":
    stocks = ["AAPL", "WMT", "TSLA", "GE", "AMZN", "DB"]

    START_DATE = "2011-01-01"
    END_DATE = "2022-01-01"

    portfolio = Portfolio(stocks=stocks, start=START_DATE, end=END_DATE)
    # portfolio.show_data()

    print(portfolio.optimize_portfolio())

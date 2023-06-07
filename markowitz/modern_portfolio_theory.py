""" A script to generate optimal portfolio.
"""


import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from alive_progress import alive_bar
import scipy.optimize as optimize

warnings.filterwarnings("ignore")


RISK_FREE = 0.045 / 252  # Based on the government bond data


# The functions outside the `Portfolio` class are used to feed into the `scipy.optimise model`
def minimise_function(weights, returns) -> np.array:
    """
    Minimisation class for the given function
    """
    return -np.array(
        statistics(weights, returns)[2]
    )  # The maximum of f(x) is the minimum of -f(x)


def statistics(weights, returns, n_days=252) -> np.array:
    """
    Calculates the required statistics for optimisation function.
    Parameters
    ----------
    weights: Portfolio weights
    returns: Log daily returns
    n_days: Number of trading days
    """

    portfolio_return = np.sum(np.dot(returns.mean(), weights.T)) * n_days
    excess_return = returns - RISK_FREE
    portfolio_volatility = np.sqrt(
        np.dot(
            weights,
            np.dot(excess_return.cov() * n_days, weights.T),
        )
    )
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            (portfolio_return - RISK_FREE * 252) / portfolio_volatility,
        ]
    )


class Portfolio:
    """
    Optimising portfolio to minimize risk.
    Input parameters:
    1. Stocks[Tickers]
    2. Start Date: The date from which the historical data is required.
    3. End Date: End point of the historical data.
    """

    NUM_TRADING_DAYS = 252
    NUM_PORTFOLIO = 50000

    def __init__(self, stocks: list, start: str, end: str) -> None:
        """
        Initalisation class of the portfolio
        """
        self.stocks = stocks
        self.start = start
        self.end = end

        # Adding the elements of the portfolio to the initalisation class
        # so that the code becomes more efficient.
        self.weights = None
        self.returns = None
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

    def calculate_return(self) -> None:
        """
        Returns the logarithmic price change of the price.
        """
        data = self.get_data_from_yahoo()
        returns = data.pct_change().dropna()
        self.returns = returns
        return returns

    def generate_portfolios(self) -> dict:
        """
        Generates random portfolios and their scatterplots.
        """
        portfolio_data = defaultdict(list)
        weights = []
        returns = self.calculate_return()
        with alive_bar(self.NUM_PORTFOLIO) as pbar:
            print("Generating portfolios \n")
            for _ in range(self.NUM_PORTFOLIO):
                weight = np.random.random(len(self.stocks))
                weight /= np.sum(weight)
                weights.append(weight)

                portfolio_return = (
                    np.sum(returns.mean() * weight) * self.NUM_TRADING_DAYS
                )
                excess_return = returns - RISK_FREE
                portfolio_data["mean"].append(portfolio_return)
                portfolio_volatility = np.sqrt(
                    np.dot(
                        weight.T,
                        np.dot(excess_return.cov() * self.NUM_TRADING_DAYS, weight),
                    )
                )
                portfolio_data["risk"].append(portfolio_volatility)
                pbar()

        plt.figure(figsize=(10, 6))
        plt.scatter(
            portfolio_data["risk"],
            portfolio_data["mean"],
            c=(np.array(portfolio_data["mean"]) - RISK_FREE * 252)
            / np.array(portfolio_data["risk"]),
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

    def optimize_portfolio(self) -> np.array:
        """
        Used to optimize the weights with respect to the sharpe ratio.
        """
        _ = self.generate_portfolios()
        returns = self.returns
        func = minimise_function
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # The weights can at the most be 1.
        bounds = tuple((0, 1) for _ in range(len(self.stocks)))

        optimum = optimize.minimize(
            fun=func,
            x0=np.array(
                self.weights[0]
            ),  # We are randomly selecting a weight for optimisation
            args=returns,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return optimum["x"].round(4)

    def display_stats(self, weights):
        """
        Displays the Sharpe Ratio, Expected return and the volatility of the
        given portfolio.
        """
        return (
            "Expected return, volatility and Sharpe ratio: ",
            statistics(weights.round(3), self.returns),
        )

    def display_and_print_portfolio(self) -> str:
        """
        Generates the point on the efficient portfolio frontier where
        the portfolio shows the optimal return and risk.
        """
        optimal = self.optimize_portfolio()
        _ = self.show_data()
        portfolio_data = self.portfolio_data
        result = {}
        for stock, optimum_weight in zip(self.stocks, optimal):
            result[stock] = optimum_weight

        print(self.display_stats(optimal))
        print("The optimum portfolio is \n")
        print(pd.DataFrame(result, index=[0]).T)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            portfolio_data["risk"],
            portfolio_data["mean"],
            c=(np.array(portfolio_data["mean"]) - RISK_FREE * 252)
            / np.array(portfolio_data["risk"]),
            marker="o",
        )
        plt.grid(True)
        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Return")
        plt.colorbar(label="Sharpe Ratio")
        plt.plot(
            statistics(optimal, self.returns)[1],
            statistics(optimal, self.returns)[0],
            "g*",
            markersize=15,
        )
        plt.show()

        # Need to add VaR model for the optimal portfolio.


if __name__ == "__main__":
    STOCKS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "IBM"]

    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=365 * 10)

    portfolio = Portfolio(
        stocks=STOCKS, start=START_DATE, end=END_DATE
    )  # We take the data of past 10 years.

    portfolio.display_and_print_portfolio()

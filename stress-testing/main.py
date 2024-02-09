"""
A script that generates optimal weights for the portfolio.
"""
from datetime import datetime, timedelta
import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf

RISK_FREE = 0.05


def generate_stress_scenarios(mean_returns, cov_matrix, num_scenarios=10000):
    """
    This function generates random cases of overall market fluctuations.
    """
    np.random.seed(42)
    return np.random.multivariate_normal(mean_returns, cov_matrix, num_scenarios)


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

    def __init__(self, stocks, expected_returns, start_date, end_date):
        """
        Initialize the PortfolioOptimizer object.

        Parameters:
            - stocks (list): List of stock symbols for portfolio construction.
            - start_date (str): Start date for collecting historical data in 'YYYY-MM-DD' format.
            - end_date (str): End date for collecting historical data in 'YYYY-MM-DD' format.
            - expected_returns (np.ndarray): Random set of returns generated for simulation.
        """
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.expected_returns = expected_returns

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

    def optimise_portfolio(self, target_return=RISK_FREE) -> np.ndarray:
        """
        Minimizes portfolio volatility subject to specified returns.

        Parameters:
            - target_return (float): Target expected return for the portfolio.

        Returns:
            optimal weights
        """
        return_data = self.calculate_returns()
        cov_matrix = np.cov(return_data, rowvar=False)
        weights = cp.Variable(len(self.stocks))
        portfolio_return = cp.sum(cp.multiply(self.expected_returns, weights))
        portfolio_volatility = cp.quad_form(weights, cov_matrix)

        # Setting the objective function
        objective = cp.Minimize(portfolio_volatility)
        constraint = [
            cp.sum(weights) == 1,
            weights >= 0,
            portfolio_return >= target_return,
        ]
        problem = cp.Problem(constraints=constraint, objective=objective)
        problem.solve()
        optimal_weights = weights.value

        portfolio_statistics = statistics(weights=optimal_weights, returns=return_data)
        return portfolio_statistics, optimal_weights


if __name__ == "__main__":
    STOCKS = ["AAPL", "MSFT", "NVDA"]
    END = datetime.today()
    START = END - timedelta(10 * 365)  # data of 10 years
    EXPECTED_RETURNS = np.random.rand(len(STOCKS))
    portfolio = Portfolio(
        stocks=STOCKS, start_date=START, end_date=END, expected_returns=EXPECTED_RETURNS
    )

    print(portfolio.optimise_portfolio())

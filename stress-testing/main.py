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

    def optimise_portfolio(self, alpha=0.05) -> np.ndarray:
        """
        Minimizes portfolio volatility subject to specified returns and stress test.

        Parameters:
            - target_returns (float): Target expected returns for the portfolio.
            - stress_test_alpha (float): Significance level for stress test (e.g., 0.05 for 5%).

        Returns:
            optimal weights
        """
        return_data = self.calculate_returns()
        cov_matrix = np.cov(return_data)
        weights = cp.Variable(len(return_data[0]))
        portfolio_volatility = cp.quad_form(weights, cov_matrix)

        # Setting the objective function
        objective_1 = cp.Minimize(portfolio_volatility)
        objective_2 = cp.Minimize(
            generate_stress_scenarios(
                np.mean(return_data.to_numpy(), axis=0),
                np.cov(return_data, rowvar=False),
            )
        )
        constraint = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(
            constraints=constraint, objective=(objective_1, objective_2)
        )
        problem.solve()
        optimal_weights = weights.value

        portfolio_returns = np.dot(self.expected_returns, optimal_weights)
        expected_shortfall = np.percentile(portfolio_returns, alpha * 100)
        return optimal_weights, expected_shortfall


if __name__ == "__main__":
    STOCKS = ["AAPL", "MSFT", "NVDA"]
    END = datetime.today()
    START = END - timedelta(10 * 365)  # data of 10 years
    EXPECTED_RETURNS = np.random.rand(len(STOCKS))
    portfolio = Portfolio(
        stocks=STOCKS, start_date=START, end_date=END, expected_returns=EXPECTED_RETURNS
    )
    data = portfolio.calculate_returns()
    stress_scenarios = generate_stress_scenarios(
        np.mean(data.to_numpy(), axis=0), np.cov(data, rowvar=False)
    )
    print(stress_scenarios)

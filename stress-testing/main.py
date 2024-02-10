"""
A script that generates optimal weights for the portfolio.
"""
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as optimize
import yfinance as yf

warnings.filterwarnings("ignore")
RISK_FREE = 0.05


def generate_stress_scenarios(mean_returns, cov_matrix, num_scenarios):
    """
    This function generates random cases of overall market fluctuations.
    """
    np.random.seed(42)
    return np.random.multivariate_normal(mean_returns, cov_matrix, num_scenarios)


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
            (portfolio_return - RISK_FREE) / portfolio_volatility,
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

    def __init__(self, stocks, start_date, end_date):
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

        # These values will be stored later
        self.returns = None

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

    def optimise_portfolio(self) -> np.array:
        """
        Used to optimize the weights with respect to the sharpe ratio.
        """

        returns = self.calculate_returns()
        self.returns = returns
        func = minimise_function
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # The weights can at the most be 1.
        bounds = tuple((0, 1) for _ in range(len(self.stocks)))
        random_weights = np.random.rand(len(self.stocks))
        random_weights /= np.sum(random_weights)
        optimum = optimize.minimize(
            fun=func,
            x0=np.array(
                random_weights
            ),  # We are randomly selecting a weight for optimisation
            args=returns,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return optimum["x"].round(4)

    def stress_test_portfolio(self, num_scenarios=10000) -> np.ndarray:
        """
        Stress tests the optimized portfolio by generating random market scenarios.

        Parameters:
            - num_scenarios (int): Number of random scenarios to generate.

        Returns:
            - Array of portfolio statistics for each scenario.
        """
        optimised_weights = self.optimise_portfolio()
        print(optimised_weights)

        # Calculate mean and covariance matrix from historical data
        returns = self.returns
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns, rowvar=False)

        # Generate stress scenarios
        stress_scenarios = generate_stress_scenarios(
            mean_returns, cov_matrix, num_scenarios
        )
        # daily_risk_free_rate = (1 + RISK_FREE) ** (1 / 252) - 1
        portfolio_volatility = np.sqrt(
            np.dot(optimised_weights, np.dot(cov_matrix, optimised_weights.T))
        )
        print(portfolio_volatility)

        # Calculate portfolio statistics for each scenario
        portfolio_statistics = defaultdict(list)
        equally_weighted_portfolio_stats = defaultdict(list)
        for i in range(num_scenarios):
            scenario_returns = stress_scenarios[i]
            portfolio_return = np.sum(np.dot(scenario_returns, optimised_weights.T))
            portfolio_statistics["Return"].append(portfolio_return)
            equally_weighted_portfolio_stats["Return"].append(np.mean(scenario_returns))

        return pd.DataFrame(portfolio_statistics), pd.DataFrame(
            equally_weighted_portfolio_stats
        )


def plot_stress_test_results(test_results_mpt, test_results_equally_weighted):
    """
    Plot histograms of stress test results for two distributions.

    Parameters:
        - test_results_mpt (pd.DataFrame): DataFrame of portfolio statistics for MPT scenarios.
        - test_results_equally_weighted (pd.DataFrame): DataFrame of portfolio statistics for equally weighted scenarios.
    """
    fig = go.Figure()

    # Add histogram trace for MPT distribution
    fig.add_trace(
        go.Histogram(
            x=test_results_mpt["Return"],
            opacity=0.7,
            name="MPT",
            marker_color="blue",
        )
    )

    # Add histogram trace for equally weighted distribution
    fig.add_trace(
        go.Histogram(
            x=test_results_equally_weighted["Return"],
            opacity=0.7,
            name="Equally Weighted",
            marker_color="orange",
        )
    )

    # Update layout
    fig.update_layout(
        title="Returns under different scenarios",
        xaxis_title="Portfolio Return",
        barmode="overlay",
    )

    fig.show()


if __name__ == "__main__":
    STOCKS = [
        "ASIANPAINT.NS",
        "BAJFINANCE.NS",
        "TITAN.NS",
        "BAJAJFINSV.NS",
        "ADANIENT.NS",
        "BRITANNIA.NS",
    ]
    END = datetime.today()
    START = END - timedelta(10 * 365)  # data of 10 years
    portfolio = Portfolio(stocks=STOCKS, start_date=START, end_date=END)
    mpt_result, equally_weigted_result = portfolio.stress_test_portfolio()
    plot_stress_test_results(mpt_result, equally_weigted_result)

    # Printing the mean and Sharpe ratio of the portfolio
    print(
        f"The average return of an equally weighted portfolio is {np.mean(equally_weigted_result['Return'])}"
    )
    print(
        f"The average return of an mpt-based portfolio is {np.mean(mpt_result['Return'])}"
    )

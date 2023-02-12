"""Implementation of Markowitz portfolio theory
"""
from collections import defaultdict
import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize


warnings.filterwarnings("ignore")

# Defining the number of trading days
NUM_TRADING_DAYS = 252

NUM_PORTFOLIO = 10000


def download_data(stocks: list, start_date, end_date) -> pd.DataFrame:
    """
    Downloads data from yfinance.
    stocks: The list of stocks
    start_date, end_date: YYYY-MM_DD format.
    """
    stock_data = {}
    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]

    return pd.DataFrame(stock_data)


def calculate_return(data: pd.DataFrame) -> np.array:
    """
    Calculates log return of the data
    """
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def generate_portfolios(returns: np.array, stocks: list) -> defaultdict:
    """
    Generates the plot of efficient portfolio frontier.
    """
    portfolio_data = defaultdict(list)

    for _ in range(NUM_PORTFOLIO):
        weight = np.random.random((len(stocks)))
        weight /= np.sum(weight)

        portfolio_data["weights"].append(weight)
        mean = np.sum(returns.mean() * weight) * NUM_TRADING_DAYS
        portfolio_data["means"].append(mean)
        std_deviation = np.sqrt(
            np.dot(weight.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weight))
        )
        portfolio_data["risks"].append(std_deviation)

    return portfolio_data


def show_portfolios(returns, volatilites):
    """Generates the plot of efficient portfolio frontier"""
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilites, returns, c=returns / np.array(volatilites), marker="o")
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()

import numpy as np
import yfinance as yf
import pandas as pd
import datetime
from scipy.stats import norm


def download_data(stock, start, end):
    """ """
    data = {}
    ticker = yf.download(stock, start, end)
    data[stock] = ticker["Adj Close"]

    return pd.DataFrame(data)


# Value at risk tomorrow


def calculate_var(position, c, mu, sigma, n=1):
    """ """

    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1 - c))

    return var


if __name__ == "__main__":
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2017, 1, 1)
    stock_data = download_data("C", start, end)
    stock_data["returns"] = np.log(stock_data["C"] / stock_data["C"].shift(1))[1:]

    # Investment
    S = 1e6

    # Confidence interval
    c = 0.95

    # Statistics of the stock data
    mu = np.mean(stock_data["returns"])
    sigma = np.std(stock_data["returns"], ddof=2)

    VaR = calculate_var(S, c, mu, sigma, 365)
    print(VaR)

import numpy as np
import yfinance as yf
import datetime
import pandas as pd
from scipy.stats import norm


def download_data(stock, start, end):
    """ """
    data = {}
    ticker = yf.download(stock, start, end)
    data[stock] = ticker["Adj Close"]

    return pd.DataFrame(data)


class VaRMonteCarlo:
    NUM_ITERATIONS = 10000

    def __init__(self, S, mu, sigma, c, n):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.NUM_ITERATIONS])

        stock_price = self.S * np.exp(
            self.n * (self.mu - 0.5 * pow(self.sigma, 2))
            + self.sigma * np.sqrt(self.n) * rand
        )

        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


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

    model = VaRMonteCarlo(1e6, mu, sigma, 0.95, 365)

    print(model.simulation())

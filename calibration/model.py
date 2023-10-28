"""
A script to calibrate the Heston's volatility model for the market data.
"""
from datetime import datetime, date, timedelta
from typing import List, Type
import pandas as pd
import numpy as np
import yfinance as yf


class Calibration:
    """
    The main class used to calibrate the Heston's stochastic model.
    """

    def __init__(self, tickers: Type[list], start: Type[datetime], end: Type[datetime]):
        """
        Initialisation of class `Calibration`
        """
        self.tickers = tickers
        self.start = start
        self.end = end

    def get_data_from_yahoo(self) -> Type[pd.DataFrame]:
        """
        Returns historical stock prices from
        """
        stock_data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data = stock.history(start=self.start, end=self.end)["Close"]
            stock_data[ticker] = data

        return pd.DataFrame(stock_data)

    def calculate_returns(self) -> Type[np.ndarray]:
        """
        Returns the log returns of the stock prices
        """
        data = self.get_data_from_yahoo()
        return np.log(data / data.shift(1)).dropna()


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "AMZN"]
    END = datetime.now()
    START = END - timedelta(365 * 5)
    model = Calibration(tickers=TICKERS, start=START, end=END)
    print(model.calculate_returns())

"""
Implementation of Moving Average Convergence Divergence(MACD)
"""
import pandas as pd
import yfinance as yf
import cufflinks as cf
import plotly.express as px

cf.go_offline()
TICKERS = ["AAPL", "AMZN", "MSFT"]
stock_data = {}

for ticker in TICKERS:
    data = yf.download(ticker, period="1mo", interval="15m")
    data.dropna(how="any", inplace=True)
    stock_data[ticker] = data


def moving_average_convergence_divergence(dataframe, a=12, b=26, c=9):
    """
    a: smaller window moving average(12 months default)
    b: larger window moving average(26 months default)
    c: signal line
    """
    data_copy = dataframe.copy()  # We dont wish to modify the existing dataframe.
    data_copy["ma_fast"] = data_copy["Adj Close"].ewm(span=a, min_periods=a).mean()
    data_copy["ma_slow"] = data_copy["Adj Close"].ewm(span=b, min_periods=b).mean()
    data_copy["macd"] = data_copy["ma_fast"] - data_copy["ma_slow"]
    data_copy["signal"] = data_copy["macd"].ewm(span=c, min_periods=c).mean()
    return data_copy.loc[:, ["macd", "signal"]]


for data in stock_data:
    stock_data[data][["MACD", "SIGNAL"]] = moving_average_convergence_divergence(
        stock_data[data]
    )

"""
Implementation of Bollinger bands and ATR
"""
import yfinance as yf
import pandas as pd
import plotly.express as px

TICKERS = ["AMZN", "GOOG", "MSFT"]
ohlcv_data = {}

for ticker in TICKERS:
    data = yf.download(ticker, period="1mo", interval="5m")
    data.dropna(inplace=True)
    ohlcv_data[ticker] = data


def bollinger_band(DF, n=14):
    df = DF.copy()
    df["MB"] = df["Adj Close"].rolling(n).mean()
    df["UB"] = df["MB"] + 2 * df["Adj Close"].rolling(n).std(ddof=0)
    df["LB"] = df["MB"] - 2 * df["Adj Close"].rolling(n).std(ddof=0)
    df["width"] - df["UB"] - df["LB"]
    return df[["MB", "UB", "LB", "width"]]


for ticker in ohlcv_data:
    ohlcv_data[ticker][["MB", "UB", "LB", "width"]] = bollinger_band(ohlcv_data[ticker])

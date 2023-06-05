"""
Implementation of Bollinger bands and ATR
"""
import yfinance as yf
import pandas as pd
import cufflinks as cf
import plotly.express as px

cf.go_offline()
TICKERS = ["AMZN", "GOOG", "MSFT"]
ohlcv_data = {}

for ticker in TICKERS:
    data = yf.download(ticker, period="1mo", interval="5m")
    data.dropna(inplace=True)
    ohlcv_data[ticker] = data


def atr(DF, n=14):
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = df["High"] - df["Adj Close"].shift(1)
    df["L-PC"] = df["Low"] - df["Adj Close"].shift(1)
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(span=n, min_periods=n).mean()
    return df["ATR"]


for ticker in ohlcv_data:
    ohlcv_data[ticker]["ATR"] = atr(ohlcv_data[ticker], n=14)

fig = px.line(ohlcv_data["MSFT"]["ATR"])
fig.show()

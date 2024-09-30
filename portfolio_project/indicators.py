"""
Implementation of technical indicators.

This module provides a collection of technical indicators commonly used in financial analysis.
Each indicator is implemented as a separate class, inheriting from a base Indicator class.
"""

from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")


class Indicator:
    """
    The base abstract class of an indicator object.

    This class serves as a template for all specific indicator classes.
    It initialises with input data and provides a structure for calculation and retrieval of results.
    """

    def __init__(self, data):
        # The implementation will work with the `pd.Series` object.
        self.data = data

        self.result = None

    def calculate(self):
        """
        Calculate the indicator values. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement calculate() method.")

    def __call__(self):
        """
        Return the calculated indicator values.
        """
        return self.result


class SMA(Indicator):
    """
    Simple Moving Average (SMA) indicator.

    Formula:
    SMA = (P1 + P2 + ... + Pn) / n

    Where:
    P = Price
    n = Number of periods
    """

    def __init__(self, data, n_period=14):
        self.n_period = n_period  # Set n_period before calling super().__init__
        super().__init__(data)  # Call the base class constructor
        self.calculate()  # Perform the calculation upon initialisation

    def calculate(self):
        # Calculate the Simple Moving Average
        self.result = self.data["Close"].rolling(window=self.n_period).mean()


class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    Formula:
    MACD Line = 12-period EMA - 26-period EMA
    Signal Line = 9-period EMA of MACD Line
    MACD Histogram = MACD Line - Signal Line

    Where:
    EMA = Exponential Moving Average
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        super().__init__(data)
        self.calculate()

    def calculate(self):
        """Calculate the MACD line, signal line, and histogram."""
        close = self.data["Close"]
        # Calculate fast and slow EMAs
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Calculate MACD histogram
        histogram = macd_line - signal_line

        self.result = pd.DataFrame(
            {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}
        )


class ATR(Indicator):
    """
    Average True Range (ATR) indicator.

    Formula:
    TR = max[(High - Low), abs(High - Close_prev), abs(Low - Close_prev)]
    ATR = SMA(TR, n)

    Where:
    TR = True Range
    SMA = Simple Moving Average
    n = Number of periods
    """

    def __init__(self, data: pd.DataFrame, period: int = 14):
        self.period = period
        super().__init__(data)
        self.calculate()

    def calculate(self):
        """Calculate the Average True Range."""
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]

        # Calculate the three different TRs
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        # Find the maximum TR
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        self.result = pd.DataFrame({"ATR": tr.rolling(window=self.period).mean()})


class StochasticOscillator(Indicator):
    """
    Stochastic Oscillator indicator.

    Formula:
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, n)

    Where:
    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    SMA = Simple Moving Average
    n = Number of periods for %D
    """

    def __init__(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        super().__init__(data)
        self.calculate()

    def calculate(self):
        """Calculate the Stochastic Oscillator (%K and %D)."""
        low_min = self.data["Low"].rolling(window=self.k_period).min()
        high_max = self.data["High"].rolling(window=self.k_period).max()

        # Calculate %K
        k = 100 * (self.data["Close"] - low_min) / (high_max - low_min)

        # Calculate %D
        d = k.rolling(window=self.d_period).mean()

        self.result = pd.DataFrame({"%K": k, "%D": d})


class WilliamsR(Indicator):
    """
    Williams %R indicator.

    Formula:
    Williams %R = (Highest High - Current Close) / (Highest High - Lowest Low) * -100

    Where:
    Highest High = highest high for the look-back period
    Lowest Low = lowest low for the look-back period
    """

    def __init__(self, data: pd.DataFrame, period: int = 14):
        self.period = period
        super().__init__(data)
        self.calculate()

    def calculate(self):
        """Calculate the Williams %R."""
        highest_high = self.data["High"].rolling(window=self.period).max()
        lowest_low = self.data["Low"].rolling(window=self.period).min()

        # Avoid division by zero by checking if highest_high equals lowest_low
        r_value_range = highest_high - lowest_low
        r_value_range[r_value_range == 0] = (
            np.nan
        )  # Set to NaN to avoid division by zero

        # Calculate Williams %R
        self.result = pd.DataFrame(
            {"Williams %R": -100 * (highest_high - self.data["Close"]) / r_value_range}
        )


if __name__ == "__main__":
    TICKERS = ["AAPL"]
    END_TEST = datetime.now()
    START_TEST = END_TEST - timedelta(days=30)  # 30 days to test the data

    # Download data
    data = yf.download(
        tickers=TICKERS,
        start=START_TEST,
        end=END_TEST,
    )

    # Calculate and print SMA
    sma = SMA(data).result
    print("Simple Moving Average:")
    print(sma[-1])

    # Assert that SMA is not None and has the expected shape
    assert sma is not None, "SMA calculation failed"
    assert isinstance(sma, pd.Series), "SMA should be a pandas Series"
    assert len(sma) == len(
        data
    ), f"SMA length ({len(sma)}) should match data length ({len(data)})"

    # Calculate and print MACD
    macd = MACD(data).result
    print("\nMoving Average Convergence Divergence:")
    print(macd)

    # Assert that MACD is not None and has the expected shape and columns
    assert macd is not None, "MACD calculation failed"
    assert isinstance(macd, pd.DataFrame), "MACD should be a pandas DataFrame"
    assert len(macd) == len(
        data
    ), f"MACD length ({len(macd)}) should match data length ({len(data)})"
    assert all(
        col in macd.columns for col in ["MACD", "Signal", "Histogram"]
    ), "MACD should have MACD, Signal, and Histogram columns"

    # Calculate and assert ATR
    atr = ATR(data).result
    assert atr is not None, "ATR calculation failed"
    assert isinstance(atr, pd.DataFrame), "ATR should be a pandas DataFrame"
    assert len(atr) == len(
        data
    ), f"ATR length ({len(atr)}) should match data length ({len(data)})"
    assert "ATR" in atr.columns, "ATR should have an ATR column"

    # Calculate and assert Stochastic Oscillator
    stoch = StochasticOscillator(data).result
    assert stoch is not None, "Stochastic Oscillator calculation failed"
    assert isinstance(
        stoch, pd.DataFrame
    ), "Stochastic Oscillator should be a pandas DataFrame"
    assert len(stoch) == len(
        data
    ), f"Stochastic Oscillator length ({len(stoch)}) should match data length ({len(data)})"
    assert all(
        col in stoch.columns for col in ["%K", "%D"]
    ), "Stochastic Oscillator should have %K and %D columns"

    # Calculate and assert Williams %R
    williams_r = WilliamsR(data).result
    assert williams_r is not None, "Williams %R calculation failed"
    assert isinstance(
        williams_r, pd.DataFrame
    ), "Williams %R should be a pandas DataFrame"
    assert len(williams_r) == len(
        data
    ), f"Williams %R length ({len(williams_r)}) should match data length ({len(data)})"
    assert (
        "Williams %R" in williams_r.columns
    ), "Williams %R should have a Williams %R column"

    print("\nAll assertions passed. Indicators are calculated correctly.")

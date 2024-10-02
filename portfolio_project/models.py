"""
Main script to train Machine Learning models based on equity data.

This module provides classes for preprocessing financial data and training
various machine learning models for trend prediction in equity markets.

Classes:
    Preprocessing: Handles data preprocessing and feature engineering.
    TrainModels: Manages the training and evaluation of machine learning models.

The script uses a variety of technical indicators and machine learning algorithms
to predict market trends.

Note: This script requires several third-party libraries including pandas,
numpy, scikit-learn, and xgboost.

Author: Harsh Parikh
Date: 2nd October 2024

"""

from datetime import timedelta, date
import warnings
from typing import Union, List, Dict
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import yfinance as yf

from indicators import (
    Indicator,
    SMA,
    MACD,
    ATR,
    StochasticOscillator,
    WilliamsR,
    OnBalanceVolume,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class Preprocessing(Indicator):
    """
    A class for preprocessing financial data and engineering features.

    This class inherits from the Indicator class and provides methods to
    calculate various technical indicators and prepare data for machine
    learning models.

    Attributes:
        data (pd.DataFrame): The input financial data.
    """

    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        """
        Initialise the Preprocessing class.

        Args:
            data (Union[pd.DataFrame, pd.Series]): Input data with daily closing prices.
                If DataFrame, it should have a 'Close' column.
                If Series, it will be treated as close prices.
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name="Close")
        super().__init__(data)
        self.data = data

    def preprocess_data(
        self,
        sma_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
        williams_period: int = 14,
    ) -> pd.DataFrame:
        """
        Preprocess the data by adding technical indicators as features.

        This method calculates various technical indicators and adds them
        as new columns to the dataset. It also calculates the trend based
        on moving averages.

        Args:
            sma_period (int): Period for Simple Moving Average
            macd_fast (int): Fast period for MACD
            macd_slow (int): Slow period for MACD
            macd_signal (int): Signal period for MACD
            atr_period (int): Period for Average True Range
            stoch_k (int): K period for Stochastic Oscillator
            stoch_d (int): D period for Stochastic Oscillator
            williams_period (int): Period for Williams %R

        Returns:
            pd.DataFrame: DataFrame with original data and added technical indicators
        """
        # Add Simple Moving Average
        self.data["SMA"] = SMA(self.data, n_period=sma_period).result

        # Add floating volatility component
        self.data["Return"] = np.log(self.data["Close"] / self.data["Close"].shift(1))
        self.data["Floating_Volatility"] = self.data["Return"].rolling(window=14).std()
        self.data.drop(["Return"], axis=1, inplace=True)

        # Add MACD
        macd_values = MACD(
            self.data,
            fast_period=macd_fast,
            slow_period=macd_slow,
            signal_period=macd_signal,
        ).result
        self.data[["MACD", "MACD_Signal", "MACD_Histogram"]] = macd_values

        # Add ATR
        if "High" not in self.data.columns:
            self.data["High"] = self.data["Close"]
        if "Low" not in self.data.columns:
            self.data["Low"] = self.data["Close"]
        self.data["ATR"] = ATR(self.data, period=atr_period).result

        # Add Stochastic Oscillator
        stoch_values = StochasticOscillator(
            self.data, k_period=stoch_k, d_period=stoch_d
        ).result
        self.data[["Stoch_K", "Stoch_D"]] = stoch_values

        # Add Williams %R
        self.data["Williams_R"] = WilliamsR(self.data, period=williams_period).result

        # Add OBV
        self.data["OBV"] = OnBalanceVolume(self.data).result

        # Calculate trend
        self._calculate_trend()

        # Remove NaN values resulting from calculations
        self.data.dropna(inplace=True)
        self.data["Trend"] = self.data["Trend"].astype(int)

        return self.data

    def _calculate_trend(self):
        """
        Calculate the trend based on moving averages.

        This method calculates 25-day and 65-day moving averages and determines
        the trend based on the following criteria:
        - The closing price must lead/lag its 25-day moving average.
        - The 25-day moving average must lead/lag to the 65-day moving average.
        - The 25-day moving average must be rising/falling for at least 5 days.
        - The 65-day moving average must be rising/falling for at least 1 day.
        """
        self.data["25 MA"] = self.data["Close"].rolling(window=25).mean()
        self.data["65 MA"] = self.data["Close"].rolling(window=65).mean()

        # Identify if the moving averages have been rising/falling
        self.data["25 MA Diff"] = self.data["25 MA"].diff(5)
        self.data["65 MA Diff"] = self.data["65 MA"].diff(1)

        # Vectorised trend calculation
        conditions = [
            (self.data["Close"] > self.data["25 MA"])
            & (self.data["25 MA"] > self.data["65 MA"])
            & (self.data["25 MA Diff"] > 0)
            & (self.data["65 MA Diff"] > 0),
            (self.data["Close"] < self.data["25 MA"])
            & (self.data["25 MA"] < self.data["65 MA"])
            & (self.data["25 MA Diff"] < 0)
            & (self.data["65 MA Diff"] < 0),
        ]
        choices = [1, -1]
        self.data["Trend"] = np.select(conditions, choices, default=0)

        # Shift trend data one step backwards
        self.data["Trend"] = self.data["Trend"].shift(-1)


class TrainModels(Preprocessing):
    """
    A class to train and evaluate machine learning models.

    This class inherits from Preprocessing and provides methods to train
    and evaluate various machine learning models on the preprocessed data.

    Attributes:
        data (pd.DataFrame): The preprocessed financial data.
        classifiers (List): A list of classifier objects to be trained and evaluated.
    """

    def __init__(self, data: pd.DataFrame, classifiers: List):
        """
        Initialise the TrainModels class.

        Args:
            data (pd.DataFrame): The preprocessed financial data.
            classifiers (List): A list of classifier objects to be trained and evaluated.
        """
        super().__init__(data)
        self.classifiers = classifiers

    def train_model(
        self, features: List[str], target: str = "Trend"
    ) -> Dict[str, float]:
        """
        Train and evaluate the machine learning models.

        This method uses time series cross-validation to train and evaluate
        the models, calculating F1 scores for each.

        Args:
            features (List[str]): List of feature column names.
            target (str): Name of the target column. Defaults to "Trend".

        Returns:
            Dict[str, float]: A dictionary of classifier names and their mean F1 scores.
        """
        X = self.data[features]
        y = self.data[target]

        f1_scores = {}
        tscv = TimeSeriesSplit(n_splits=5)

        for clf in self.classifiers:
            print(f"\nEvaluating {clf.__class__.__name__} with TimeSeriesSplit:")
            model_pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ovr", OneVsRestClassifier(clf)),
                ]
            )

            iterative_f1_scores = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model_pipeline.fit(X_train, y_train)
                y_pred = model_pipeline.predict(X_test)

                current_f1_score = f1_score(y_test, y_pred, average="weighted")
                iterative_f1_scores.append(current_f1_score)

            f1_scores[clf.__class__.__name__] = np.mean(iterative_f1_scores)
            print(classification_report(y_test, y_pred))

        print(f"Model with the highest F1 score: {max(f1_scores, key=f1_scores.get)}")

        return f1_scores


if __name__ == "__main__":

    END_TRAINING = date(2010, 1, 1)
    START_TRAINING = END_TRAINING - timedelta(
        days=365 * 10 + 2
    )  # 10 years, accounting for 2 leap years

    aapl_data = yf.download("AAPL", start=START_TRAINING, end=END_TRAINING)

    preprocessor = Preprocessing(aapl_data)
    processed_data = preprocessor.preprocess_data()

    classifiers = [
        LogisticRegression(random_state=42),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        SVC(random_state=42),
        XGBClassifier(random_state=42),
    ]

    trainer = TrainModels(processed_data, classifiers)
    features = [col for col in processed_data.columns if col != "Trend"]
    f1_scores = trainer.train_model(features)

    print(f1_scores)

"""
CAPM Implementation
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Market risk free rate(Interest obtained by a government bond/ treasury bill)
RISK_FREE_RATE = 0.05

# Since we are considering montly returns.
NUM_MONTHS_IN_YEAR = 12


class CAPM:
    """
    Implementation of CAPM pricing model.
    """

    def __init__(self, stocks: list, start_date: str, end_date: str) -> None:
        """
        Initialisation class of the CAPM model.
        ---------------------------------------
        Input parameters:
        1. Stocks: The list of stocks to be invested in.
        2. Start date: YYYY-MM-DD Format.
        3. End Date: YYYY-MM-DD Format.
        """
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self) -> pd.DataFrame:
        """
        Downloads data from the yahoo finance server. Returns pandas dataframe
        object
        """

        data = {}

        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker["Adj Close"]  # We select Adj closing price because
            # it accounts factors such as dividends
            # stock splits etc.

        return pd.DataFrame(data)

    def initialise(self):
        """
        Function that generates the required data
        """
        stock_data = self.download_data().resample("M").last()

        self.data = pd.DataFrame(
            {
                "s_adjclose": stock_data[self.stocks[0]],
                "m_adjclode": stock_data[self.stocks[1]],
            }
        )

        # Calculating logarithmic monthly returns

        self.data[["s_returns", "m_returns"]] = np.log(
            self.data[["s_adjclose", "m_adjclode"]]
            / self.data[["s_adjclose", "m_adjclode"]].shift(1)
        )

        self.data = self.data[1:]

    def calculate_beta(self):
        """
        Calculation of Beta.
        Formula: beta = Cov(r_m, r_a)/Var(r_m)
        """
        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])

        # Calculating beta
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

        return beta

    def regression(self):
        """
        Fits the CAPM line.
        """
        beta, alpha = np.polyfit(self.data["m_returns"], self.data["s_returns"], deg=1)

        expected_return = RISK_FREE_RATE + beta * (
            self.data["m_returns"].mean() * NUM_MONTHS_IN_YEAR - RISK_FREE_RATE
        )
        self.plot_regression(alpha=alpha, beta=beta)
        return expected_return, beta, alpha

    def plot_regression(self, alpha, beta):
        """
        Plots the CAPM line
        """
        _, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(
            self.data["m_returns"], self.data["s_returns"], label="Data Points"
        )
        axis.plot(
            self.data["m_returns"],
            beta * self.data["m_returns"] + alpha,
            color="red",
            label="CAPM Line",
        )
        plt.title("Capital Asset Pricing Model, finding alphas and betas")
        plt.xlabel("Market Return $r_m$", fontsize=18)
        plt.ylabel("Stock Return $r_a$")
        plt.text(0.08, 0.05, r"$r_a = \beta * r_m + \alpha$", fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    START_DATE = "2010-01-01"
    END_DATE = "2023-01-01"

    capm = CAPM(["IBM", "^GSPC"], START_DATE, END_DATE)
    capm.initialise()
    print(capm.calculate_beta())
    print(capm.regression())

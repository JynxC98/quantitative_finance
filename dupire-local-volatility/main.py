"""
Script for generating volatility surface graphs.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
import yfinance as yf
from volatily import Option

warnings.filterwarnings("ignore")


class VolatilitySurface:
    """
    Class for generating and plotting the implied volatility surface of a stock.
    """

    def __init__(self, ticker, risk_free_rate):
        """
        Initializes the VolatilitySurface instance.

        Parameters:
        - ticker (str): The ticker symbol of the company.
        - risk_free_rate (float): The risk-free interest rate.
        """
        self._ticker = ticker
        self._risk_free_rate = risk_free_rate
        self._spot = None
        self._ttm_grid = None
        self._strike_grid = None

    def get_spot_price(self):
        """
        Returns the spot price of the stock.
        """
        stock = yf.Ticker(self._ticker)
        self._spot = stock.history(period="1d")["Close"].iloc[-1]
        return self._spot

    def get_strike_price_pivot_table(self, option_type="call", min_rows=30):
        """
        Returns a pivot table with time to maturity as rows, strike prices as columns,
        and option prices as values.

        Parameters:
        - option_type (str): "call" for call options, "put" for put options.
        - min_rows (int): Minimum number of rows required to consider the option chain.

        Returns:
        - DataFrame: Pivot table with TTM as rows, strike prices as columns, and
          option prices as values.
        """
        if option_type not in ("call", "put"):
            raise ValueError(
                "Invalid option type selection. Select one from `call` or `put`."
            )
        company_data = yf.Ticker(self._ticker)

        maturities = (
            company_data.options
        )  # This code fetches all the available option contract based on their expiration dates

        strikes = set(
            company_data.option_chain(maturities[0]).calls["strike"]
        )  # We store the first set of strike prices for evaluation.
        spot_price = self.get_spot_price()

        options_data = []

        for maturity in maturities:
            mat_time = (pd.to_datetime(maturity) - pd.Timestamp.today()).days / 252
            if (
                mat_time <= 0.1 or mat_time >= 2
            ):  # Filtering the data based on time to maturity.
                continue

            data = company_data.option_chain(maturity)
            data = data.calls if option_type == "call" else data.puts

            if data.shape[0] < min_rows:
                continue

            strikes = strikes.intersection(data["strike"])
            filtered_strikes = {
                strike
                for strike in strikes
                if 0.95 * spot_price <= strike <= 1.1 * spot_price
            }
            filtered_data = data[data["strike"].isin(filtered_strikes)]

            filtered_data["TTM"] = mat_time
            options_data.append(filtered_data)

        combined_data = pd.concat(options_data, ignore_index=True)
        pivot_table = combined_data.pivot_table(
            index="TTM", columns="strike", values="lastPrice", fill_value=0
        )

        return pivot_table

    def generate_implied_volatility_surface(self):
        """
        Generates the implied volatility surface.
        """
        option_data = self.get_strike_price_pivot_table()
        strikes = option_data.columns.values
        time_to_mat = option_data.index.values
        option_prices = option_data.values

        implied_vols = np.zeros(option_prices.shape)

        for i, time in enumerate(time_to_mat):
            for j, strike in enumerate(strikes):
                option = Option(
                    spot=self._spot,
                    strike=strike,
                    risk_free=self._risk_free_rate,
                    maturity=time,
                )
                implied_vols[i, j] = option.implied_volatility(
                    actual_price=option_prices[i, j]
                )

        ttm_grid, strike_grid = np.meshgrid(time_to_mat, strikes)
        self._ttm_grid = ttm_grid
        self._strike_grid = strike_grid

        iv_surface = RectBivariateSpline(time_to_mat, strikes, implied_vols)
        iv_values = iv_surface.ev(ttm_grid.ravel(), strike_grid.ravel()).reshape(
            ttm_grid.shape
        )
        iv_values[iv_values < 0] = (
            0  # Using this to filter out the negative implied volatility generated due to interpolation.
        )

        return iv_values

    def plot_imp_vol_surface(self):
        """
        Plots the implied volatility surface.
        """
        iv_values = self.generate_implied_volatility_surface()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        surf = ax.plot_surface(
            self._ttm_grid,
            np.log(self._strike_grid / self._spot),
            iv_values,
            cmap="viridis",
            edgecolor="none",
            antialiased=True,
        )

        # Labels and title
        ax.set_xlabel("Time to Maturity", fontsize=12, labelpad=10)
        ax.set_ylabel("Moneyness (log scale)", fontsize=12, labelpad=10)
        ax.set_zlabel("Implied Volatility", fontsize=12, labelpad=10)
        ax.set_title("Implied Volatility Surface", fontsize=16, pad=20)

        # Tick parameters
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="z", which="major", labelsize=10)

        # The view angle for better visual
        ax.view_init(elev=30, azim=120)

        # Adding grid for better readability
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        cbar.set_label("Implied Volatility", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        plt.show()


if __name__ == "__main__":
    RISK_FREE_RATE = 0.0467  # Fed rate
    TICKER = "AAPL"

    surface = VolatilitySurface(ticker=TICKER, risk_free_rate=RISK_FREE_RATE)
    surface.plot_imp_vol_surface()

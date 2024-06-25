"""
Script for generating volatility surface graphs.
"""

from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
import yfinance as yf
from option_properties import Option

warnings.filterwarnings("ignore")


class VolatilitySurface:
    """
    A class to generate and plot the implied volatility surface for a given stock.

    This class provides methods to:
    - Retrieve the spot price of the stock.
    - Generate a pivot table of option prices with strike prices as columns and time to maturity (TTM) as rows.
    - Calculate the implied volatility for a range of strike prices and maturities.
    - Plot the implied volatility surface using a 3D plot.

    Attributes:
    ----------
    _ticker : str
        The ticker symbol of the stock.
    _risk_free_rate : float
        The risk-free interest rate used in calculations.
    _spot : float
        The current spot price of the stock.
    _ttm_grid : ndarray
        A grid of time to maturity values.
    _strike_grid : ndarray
        A grid of strike prices.

    Methods:
    -------
    get_spot_price():
        Retrieves the current spot price of the stock.

    get_strike_price_pivot_table(option_type="call", min_rows=30):
        Creates a pivot table of option prices with TTM as rows and strike prices as columns.

    generate_implied_volatility_surface():
        Calculates the implied volatility surface based on option prices.

    plot_imp_vol_surface():
        Plots the implied volatility surface using a 3D plot.
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

        # These values are to be calculated later.
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

    def get_strike_price_pivot_table(
        self,
        option_type="call",
        maturity_min=0.1,
        maturity_max=2,
        moneyness_min=0.95,
        moneyness_max=1.5,
    ):
        """
        Generate a pivot table of option strike prices for a given ticker.

        This function fetches option data for a specified stock ticker and creates a pivot table
        of strike prices across different maturities. It filters the data based on time to maturity
        and moneyness (strike price relative to spot price).

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
            maturity_min (float, optional): Minimum time to maturity in years. Defaults to 0.1.
            maturity_max (float, optional): Maximum time to maturity in years. Defaults to 2.
            moneyness_min (float, optional): Minimum moneyness (strike/spot) to consider. Defaults to 0.95.
            moneyness_max (float, optional): Maximum moneyness (strike/spot) to consider. Defaults to 1.2.

        Returns:
            pd.DataFrame: A pivot table with time to maturity as index, strike prices as columns,
                        and option last prices as values. Strikes not available for a particular
                        maturity are filled with 0.
        """
        if option_type not in ("call", "put"):
            raise ValueError(
                "Invalid option type selection. Select one from `call` or `put`."
            )
        option_data = yf.Ticker(self._ticker)
        today = pd.Timestamp.today()
        valid_maturities = [
            mat
            for mat in option_data.options
            if maturity_min < (pd.to_datetime(mat) - today).days / 252 < maturity_max
        ]

        spot_price = self.get_spot_price()
        strikes_freq = defaultdict(int)
        all_data = []
        for maturity in valid_maturities:
            chain = option_data.option_chain(maturity).calls
            ttm = (pd.to_datetime(maturity) - today).days / 252

            valid_strikes = chain[
                (chain["strike"] >= moneyness_min * spot_price)
                & (chain["strike"] <= moneyness_max * spot_price)
            ]

            for strike in valid_strikes["strike"]:
                strikes_freq[strike] += 1

            valid_strikes["TTM"] = ttm
            all_data.append(valid_strikes[["strike", "lastPrice", "TTM"]])

        common_strikes = {
            strike
            for strike, freq in strikes_freq.items()
            if freq == len(valid_maturities)
        }

        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data[combined_data["strike"].isin(common_strikes)]

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
    TICKER = "MSFT"

    surface = VolatilitySurface(ticker=TICKER, risk_free_rate=RISK_FREE_RATE)
    surface.plot_imp_vol_surface()

"""
A script to calculate the local volatility of an underlying based on
Dupire's formula

Author: Harsh Parikh

Date: 01-02-2025
"""

from typing import Type, Tuple
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


from fetch_data import get_strike_price_pivot_table


class LocalVol:
    """
    Computes the local volatility surface using the Dupire framework as
    described in Jim Gatheral's *The Volatility Surface*.

    This class supports two approaches for computing the local volatility:
    (1) a custom Python implementation, and
    (2) an implementation based on QuantLib.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol for which the option data is used.

    moneyness_min : float, optional (default=0.8)
        Minimum moneyness threshold for filtering option data.

    moneyness_max : float, optional (default=1.3)
        Maximum moneyness threshold for filtering option data.

    maturity_min : float, optional (default=0.1)
        Minimum time-to-maturity (in years) considered for the option data.

    maturity_max : float, optional (default=2.0)
        Maximum time-to-maturity (in years) considered for the option data.

    method : str, optional (default="python")
        Method used to compute the local volatility surface.
        Available options:
        - "python"   : custom Python implementation
        - "quantlib" : QuantLib-based implementation
    """

    def __init__(
        self,
        ticker,
        moneyness_min=0.8,
        moneyness_max=1.3,
        maturity_min=0.1,
        maturity_max=2,
        method="python",
    ):
        """
        The initialization method for the `LocalVol` class.
        """


def plot_volatility_surface(
    vol_surface: Type[np.ndarray],
    maturities: Type[np.ndarray],
    strikes: Type[np.ndarray],
) -> None:
    """
    Plot the local volatility surface in 3D using interpolation.
    """

    # Create meshgrid
    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")

    # Create an interpolator
    lv_surface = RectBivariateSpline(maturities, strikes, vol_surface)

    # Evaluate on the grid
    lv_values = lv_surface.ev(T_grid, K_grid)

    # Check for NaNs or Infs
    if np.any(np.isnan(lv_values)) or np.any(np.isinf(lv_values)):
        print(
            "Warning: lv_values contains NaNs or Infs. Plot may not display correctly."
        )

    # Plot the surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        T_grid, K_grid, lv_values, cmap="viridis", edgecolor="k", alpha=0.8
    )

    # Set labels and title
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (T)")
    ax.set_zlabel("Local Volatility")
    ax.set_title("Local Volatility Surface")

    plt.colorbar(surf)
    plt.show(block=True)  # Use block=True to keep the plot window open


def main() -> None:
    """
    The main execution block
    """
    # Example usage
    TICKER = "AAPL"
    RATE = 0.05
    DIV = 0.02

    _, option_data = get_strike_price_pivot_table(ticker=TICKER)

    mat = option_data.index.values
    strikes = option_data.columns.values
    option_prices = option_data.values

    local_vol, maturities, strikes = calculate_local_volatility(
        maturities=mat,
        strikes=strikes,
        option_prices=option_prices,
        r=RATE,
        q=DIV,
        grid_points=20,
    )

    local_vol[np.isnan(local_vol)] = 0

    # print(local_vol)

    plot_volatility_surface(local_vol, maturities, strikes)


if __name__ == "__main__":
    # Example usage
    main()

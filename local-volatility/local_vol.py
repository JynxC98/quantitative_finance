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


def calculate_local_volatility(
    maturities: Type[np.ndarray],
    strikes: Type[np.ndarray],
    option_prices: Type[np.ndarray],
    r: Type[float],
    q: Type[float],
    grid_points: Type[float] = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the local volatility surface using Dupire's formula.

    Parameters:
        ticker (str): The ticker symbol for the underlying asset.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        grid_points (int): Number of points for interpolation grid.

    Returns:
        local_vol_surface (np.array): 2D array of local volatilities.
        maturities (np.array): Array of maturities.
        strikes (np.array): Array of strike prices.
    """

    # Create a finer grid for interpolation
    interpolated_maturities = np.linspace(
        maturities.min(), maturities.max(), grid_points + 1
    )
    interpolated_strikes = np.linspace(strikes.min(), strikes.max(), grid_points + 1)

    # Interpolate option prices using RectBivariateSpline
    spline = RectBivariateSpline(maturities, strikes, option_prices)

    # Create meshgrid for interpolated maturities and strikes
    T_grid, K_grid = np.meshgrid(
        interpolated_maturities, interpolated_strikes, indexing="ij"
    )

    # Evaluate interpolated option prices on the grid
    interpolated_option_prices = spline.ev(T_grid, K_grid)

    # Calculating the time difference
    dt = (maturities.max() - maturities.min()) / grid_points

    # Calculating the spatial grid
    dK = (strikes.max() - strikes.min()) / grid_points

    # Compute derivatives using finite differences
    dC_dt = np.gradient(interpolated_option_prices, dt, axis=0)
    dC_dK = np.gradient(interpolated_option_prices, dK, axis=1)
    d2C_dK2 = np.gradient(dC_dK, dK, axis=1)

    # Compute the numerator of Dupire's formula
    numerator = (
        dC_dt + q * interpolated_option_prices + (r - q) * interpolated_strikes * dC_dK
    )

    # Compute the denominator, ensuring no division by zero
    denominator = 0.5 * (interpolated_strikes**2) * d2C_dK2
    denominator[denominator == 0] = np.nan  # Avoid division by zero

    # Compute local volatility
    local_vol_data = np.sqrt(numerator / denominator)

    return local_vol_data, interpolated_maturities, interpolated_strikes


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

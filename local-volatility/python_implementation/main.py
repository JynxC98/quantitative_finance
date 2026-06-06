"""
A script to compute the local volatility based on Dupire's formula.

Author: Harsh Parikh
Date: 4th January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fetch_data import get_strike_price_pivot_table


def local_volatility(T, K, C, r, q):
    """
    Calculate local volatility using the finite difference method.

    Parameters:
        T (np.array): Array of maturities.
        K (np.array): Array of strike prices.
        C (np.array): 2D array of call option prices (rows: maturities, columns: strikes).
        r (float): Risk-free interest rate.
        q (float): Dividend yield.

    Returns:
        sigma (np.array): 2D array of local volatilities.
    """
    # Ensure inputs are numpy arrays
    T = np.array(T)
    K = np.array(K)
    C = np.array(C)

    # Initialize local volatility matrix
    sigma = np.zeros_like(C)

    # Step sizes for finite differences
    dT = np.diff(T)
    dK = np.diff(K)

    # Loop over maturities and strikes
    for i in range(1, len(T) - 1):
        for j in range(1, len(K) - 1):
            # Partial derivatives using central finite differences
            dC_dT = (C[i + 1, j] - C[i - 1, j]) / (2 * dT[i - 1])
            dC_dK = (C[i, j + 1] - C[i, j - 1]) / (2 * dK[j - 1])
            d2C_dK2 = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / (dK[j - 1] ** 2)

            # Avoid division by zero
            if d2C_dK2 == 0:
                sigma[i, j] = np.nan
                continue

            # Local volatility formula
            numerator = dC_dT + (r - q) * K[j] * dC_dK - r * C[i, j]
            denominator = 0.5 * K[j] ** 2 * d2C_dK2
            sigma[i, j] = np.sqrt(2 * numerator / denominator)

    return sigma


# Example usage
if __name__ == "__main__":
    # Example data
    _, option_data = get_strike_price_pivot_table("AAPL")
    T = option_data.index.values
    K = option_data.columns.values
    C = option_data.values

    r = 0.05  # Risk-free rate
    q = 0.02  # Dividend yield

    # Calculate local volatility
    sigma = local_volatility(T, K, C, r, q)

    # Print results
    print("Local Volatility Surface:")
    print(sigma)

    # Create a meshgrid for 3D plotting
    T_grid, K_grid = np.meshgrid(T, K, indexing="ij")

    # Plot the local volatility surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T_grid, K_grid, sigma, cmap="viridis", edgecolor="none")

    # Add labels and title
    ax.set_xlabel("Maturity (T)")
    ax.set_ylabel("Strike (K)")
    ax.set_zlabel("Local Volatility (σ)")
    ax.set_title("Local Volatility Surface")

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()

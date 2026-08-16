"""
This script stores the function for calculating the value of a geometric
Asian option using the analytical formula.

Author: Harsh Parikh
"""

import numpy as np
import math
from numba import njit
import warnings

warnings.filterwarnings("ignore")


@njit()
def geometric_asian_option(
    *,
    spots: np.ndarray,
    weights: np.ndarray,
    vols: np.ndarray,
    corr_matrix: np.ndarray,
    strike: float,
    r: float,
    T: float,
    div_yields: np.ndarray,
    isCall: bool = True,
):
    """
    This function calculates the value of a geometric Asian option using
    the analytical closed-form solution for multi-asset geometric Brownian
    motion. The solution is based on the framework from:

    https://www.jstor.org/stable/2331213

    The geometric average of multiple correlated assets follows a log-normal
    distribution, allowing the option to be priced using a modified
    Black-Scholes formula that accounts for the correlation structure and
    dividend yields.

    Args:
        spots: A 1D numpy array of initial asset prices (S_i) at time 0.
               The length of this array (N) determines the number of assets.
        weights: A 1D numpy array of weights (w_i) for each asset in the
                 geometric average, corresponding to the order in `spots`.
                 These weights must sum to 1.
        vols: A 1D numpy array of annualized volatilities (sigma_i) for
              each asset.
        corr_matrix: An N x N numpy array containing the correlation
                     coefficients (rho_ij) between the assets. Must be
                     positive semi-definite with diagonal elements equal to 1.
        strike: The strike price (K) of the option.
        r: The risk-free interest rate (annualized, as a decimal, e.g., 0.05 for 5%).
        T: The time to maturity of the option in years.
        div_yields: 1D numpy array of continuous dividend yields (q_i)
                    for each asset.
        isCall: A boolean flag. If True, prices a call option.
                If False, prices a put option.

    Returns:
        The present value of the geometric Asian option as a float.
    """
    # Performing rudimentary check
    assert (
        np.all(np.linalg.eigvals(corr_matrix)) > 0,
        "The underlying correlation matrix should be positive semidefinite",
    )

    # Calculating the first moment of the underlying option basket
    agg_mean = np.dot(weights, ((np.log(spots) + (r - div_yields - 0.5 * vols**2) * T)))

    # Calculating the second moment of the underlying option baskets
    vol_outer = np.outer(vols, vols)
    cov_matrix = corr_matrix * vol_outer
    agg_variance = np.dot(weights, np.dot(cov_matrix, weights)) * T

    # Evaluating terms for the closed form solution
    forward = np.exp(agg_mean + 0.5 * agg_variance)

    # Calculating d1
    d1 = (np.log(forward / strike) + 0.5 * agg_variance) / np.sqrt(agg_variance)

    # Calculating d2
    d2 = d1 - np.sqrt(agg_variance)

    sign = 1 if isCall else -1

    # Generating the cdf function using lambda
    cdf = lambda x: 0.5 * math.erfc(-x / np.sqrt(2))

    return sign * np.exp(-r * T) * (forward * cdf(sign * d1) - strike * cdf(sign * d2))


def _test_case_1():
    """
    Test Case 1: Two-asset geometric Asian call option with correlation.
    """
    print("=" * 50)
    print("TEST CASE 1: Two-Asset Call Option")
    print("=" * 50)

    # Parameters
    spots = np.array([100.0, 100.0])
    weights = np.array([0.5, 0.5])
    vols = np.array([0.2, 0.3])
    corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    strike = 100.0
    r = 0.05
    T = 1.0
    div_yields = np.array([0.02, 0.01])

    # Price call option
    price = geometric_asian_option(
        spots=spots,
        weights=weights,
        vols=vols,
        corr_matrix=corr_matrix,
        strike=strike,
        r=r,
        T=T,
        div_yields=div_yields,
        isCall=True,
    )

    print(f"Parameters:")
    print(f"  Spots: {spots}")
    print(f"  Weights: {weights}")
    print(f"  Vols: {vols}")
    print(f"  Correlation: 0.5")
    print(f"  Strike: {strike}")
    print(f"  r: {r}, T: {T}")
    print(f"  Div yields: {div_yields}")
    print(f"\nCall Price: {price:.6f}")

    return price


def _test_case_2():
    """
    Test Case 2: Three-asset geometric Asian put option with different weights.
    """
    print("\n" + "=" * 50)
    print("TEST CASE 2: Three-Asset Put Option")
    print("=" * 50)

    # Parameters
    spots = np.array([100.0, 120.0, 90.0])
    weights = np.array([0.4, 0.3, 0.3])
    vols = np.array([0.2, 0.25, 0.35])
    corr_matrix = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    strike = 105.0
    r = 0.04
    T = 2.0
    div_yields = np.array([0.01, 0.02, 0.015])

    # Price put option
    price = geometric_asian_option(
        spots=spots,
        weights=weights,
        vols=vols,
        corr_matrix=corr_matrix,
        strike=strike,
        r=r,
        T=T,
        div_yields=div_yields,
        isCall=False,
    )

    print(f"Parameters:")
    print(f"  Spots: {spots}")
    print(f"  Weights: {weights}")
    print(f"  Vols: {vols}")
    print(f"  Correlation matrix:")
    print(corr_matrix)
    print(f"  Strike: {strike}")
    print(f"  r: {r}, T: {T}")
    print(f"  Div yields: {div_yields}")
    print(f"\nPut Price: {price:.6f}")

    return price


def main():
    _test_case_1()
    _test_case_2()


if __name__ == "__main__":
    main()

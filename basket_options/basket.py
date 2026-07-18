"""
An implementation of the basket options using Monte-Carlo simulations.

Author: Harsh Parikh
"""

from numpy.typing import NDArray
import numpy as np
from scipy.linalg import cholesky


def calculate_basket_option_price(
    spot_arr: NDArray[np.float64],
    strike: float,
    vols: NDArray[np.float64],
    weights: NDArray[np.float64],
    corr_mat: NDArray[np.float64],
    r: float,
    T: float,
    isCall: bool = True,
    M: int = 50000,
) -> float:
    """
    This function serves as a vanilla implementation to calculate the
    theoretical value of the basket options. The input parameters are as
    follows:

    Input Parameters
    ----------------
    spot_arr: The array of spot prices.
    strike: The strike price of the contract.
    vols: The array of asset volatilities.
    weights: The weights of the assets in the portfolio.
    corr_mat: The correlation matrix for the asset evolution.
    r: The risk free rate
    T: Time to maturity
    isCall: True for call and false otherwise
    M: The number of Monte-Carlo paths.
    """

    # Assigning a seed value for reproducibility
    np.random.seed(42)

    # Calculating the number of assets in the portfolio
    N = len(spot_arr)

    # Storing the option payoffs
    option_payoffs = np.zeros(M)

    # Fetching the correlated Brownian increments
    cholesky_mat = cholesky(corr_mat, lower=True)  # We need an implementation of a
    # lower triangular matrix for correlated Brownian motion.

    dW = np.random.standard_normal(size=(M, N))

    dW_correlated = np.dot(dW, cholesky_mat.T)  # This code will generate a vector of
    # M x N, indicating N correlated Brownian increments for M paths

    # Multiplying the weights to the asset prices for easy computation.
    spot_arr_weighted = spot_arr * weights

    # Calculating the drift of the process
    drift = (r - 0.5 * vols**2) * T

    sqrt_T = np.sqrt(T)

    diffusion = vols * sqrt_T * dW_correlated  # M x N

    basket_values = np.sum(
        spot_arr_weighted * np.exp(drift + diffusion), axis=1
    )  # M x N

    if isCall:
        option_payoffs = np.maximum(basket_values - strike, 0.0)
    else:
        option_payoffs = np.maximum(strike - basket_values, 0.0)

    # Discounting the expected payoff under the risk-neutral measure
    payoff = np.exp(-r * T) * np.mean(option_payoffs)

    std_err = np.std(option_payoffs) / np.sqrt(M)

    return {
        "mean_price": payoff,
        "left_ci": payoff - 1.96 * std_err,
        "right_ci": payoff + 1.96 * std_err,
        "std_error": std_err,
    }


if __name__ == "__main__":
    value = calculate_basket_option_price(
        spot_arr=np.array([100, 120]),
        strike=120,
        vols=np.array([0.25] * 2),
        weights=np.array([0.5] * 2),
        corr_mat=np.array([[1, -0.65], [-0.65, 1]]),
        r=0.0405,
        T=1.0,
    )
    print(value)

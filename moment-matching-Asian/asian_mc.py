"""
This script stores the implementation of the arithmetic Asian option valuation
using geometric Asian option as a control variate.

Author: Harsh Parikh
"""

import numpy as np
from numba import jit, prange
from geometric_asian import geometric_asian_option


@jit(nopython=True, cache=True)
def _arithmetic_asian_worker(
    spots: np.ndarray,
    weights: np.ndarray,
    vols: np.ndarray,
    L: np.ndarray,
    strike: float,
    r: float,
    T: float,
    div_yields: np.ndarray,
    isCall: bool,
    M: int,
    N: int,
) -> tuple:
    """
    Internal worker function for Monte Carlo simulation.
    Separated from main function to handle RNG state correctly.
    """
    d = len(spots)
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    drift = (r - div_yields - 0.5 * vols**2) * dt

    sum_arith_payoffs = 0.0
    sum_geo_payoffs = 0.0

    for _ in prange(M):
        prices = spots.copy()
        sum_prices = np.zeros(d)
        sum_log_prices = np.zeros(d)

        for _ in range(N):
            Z = np.random.normal(0, 1, d)
            correlated_Z = L @ Z

            prices = prices * np.exp(drift + vols * sqrt_dt * correlated_Z)

            sum_prices += prices
            sum_log_prices += np.log(prices)

        arith_avg = np.dot(weights, sum_prices / N)
        geo_avg = np.exp(np.dot(weights, sum_log_prices / N))

        if isCall:
            arith_payoff = max(arith_avg - strike, 0.0)
            geo_payoff = max(geo_avg - strike, 0.0)
        else:
            arith_payoff = max(strike - arith_avg, 0.0)
            geo_payoff = max(strike - geo_avg, 0.0)

        sum_arith_payoffs += arith_payoff
        sum_geo_payoffs += geo_payoff

    return sum_arith_payoffs, sum_geo_payoffs


def arithmetic_asian_mc(
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
    M: int = 1000000,
    N: int = 252,
):
    """
    Calculate the price of an arithmetic Asian option using Monte Carlo simulation
    with the geometric Asian option as a control variate.

    This function simulates the path of multiple correlated assets under geometric
    Brownian motion and computes the payoff based on the arithmetic average of
    asset prices over the simulation horizon. The option value is estimated using
    the control variate technique with the analytically known geometric Asian
    option price to reduce variance and improve convergence.

    Args:
        spots: A 1D numpy array of initial asset prices (S_i) at time 0.
               The length of this array (d) determines the number of assets.
        weights: A 1D numpy array of weights (w_i) for each asset in the
                 arithmetic average, corresponding to the order in `spots`.
                 These weights must sum to 1.
        vols: A 1D numpy array of annualized volatilities (sigma_i) for
              each asset.
        corr_matrix: A d x d numpy array containing the correlation
                     coefficients (rho_ij) between the assets. Must be
                     positive semi-definite with diagonal elements equal to 1.
        strike: The strike price (K) of the option.
        r: The risk-free interest rate (annualized, as a decimal, e.g., 0.05 for 5%).
        T: The time to maturity of the option in years.
        div_yields: A 1D numpy array of continuous dividend yields (q_i)
                    for each asset.
        isCall: A boolean flag. If True, prices a call option.
                If False, prices a put option. Default is True.
        M: The number of Monte Carlo simulation paths. Default is 1,000,000.
        N: The number of time steps per path. Default is 252 (trading days in a year).
        seed: The seed value for the random number generator

    Returns:
        The estimated present value of the arithmetic Asian option as a float.

    References:
        - Boyle, P., Broadie, M., & Glasserman, P. (1997). Monte Carlo methods
          for security pricing. Journal of Economic Dynamics and Control, 21(8-9),
          1267-1321.
        - Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
          Springer.
    """
    d = len(spots)

    # Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(corr_matrix)

    # Compute geometric Asian option price (analytical control variate)
    geo_price = geometric_asian_option(
        spots=spots,
        weights=weights,
        vols=vols,
        corr_matrix=corr_matrix,
        strike=strike,
        r=r,
        T=T,
        div_yields=div_yields,
        isCall=isCall,
    )

    # Call worker function with explicit parallelization
    sum_arith_payoffs, sum_geo_payoffs = _arithmetic_asian_worker(
        spots=spots,
        weights=weights,
        vols=vols,
        L=L,
        strike=strike,
        r=r,
        T=T,
        div_yields=div_yields,
        isCall=isCall,
        M=M,
        N=N,
    )

    # Compute discounted expected payoffs
    arith_mc_price = np.exp(-r * T) * (sum_arith_payoffs / M)
    geo_mc_price = np.exp(-r * T) * (sum_geo_payoffs / M)

    # Control variate adjustment
    # V_cv = V_geo_analytic + (V_arith_mc - V_geo_mc)
    cv_price = geo_price + (arith_mc_price - geo_mc_price)

    return cv_price


if __name__ == "__main__":

    spots = np.array([100.0, 100.0])
    weights = np.array([0.5, 0.5])
    vols = np.array([0.2, 0.3])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    div = np.array([0.02, 0.01])

    price = arithmetic_asian_mc(
        spots=spots,
        weights=weights,
        vols=vols,
        corr_matrix=corr,
        strike=100.0,
        r=0.05,
        T=1.0,
        div_yields=div,
        M=100000,
        N=252,
        isCall=True,
    )
    print(f"Arithmetic Asian option price: {price:.6f}")

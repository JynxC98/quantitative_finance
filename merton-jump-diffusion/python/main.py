"""
This script provides a numerical simulation of the Merton Jump-Diffusion (MJD) model, 
a stochastic process that extends the standard Geometric Brownian Motion (GBM) by 
incorporating random jumps. The implementation leverages NumPy for efficient 
computation and Numba JIT compilation for performance.

Author
------
Harsh Parikh  

Date
----
8th Feb 2025

Reference
---------
Matsuda, M. (n.d.). Introduction to Merton Jump Diffusion. 
Retrieved from: https://www.maxmatsuda.com/Papers/Intro/Intro%20to%20MJD%20Matsuda.pdf
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def jump_diffusion(spot, strike, sigma, r, T, lambda_, mu_j, sigma_j, M=5000, N=5000):
    """
    Simulate the Merton Jump-Diffusion process and estimate a European option price.

    Parameters
    ----------
    spot : float
        Initial asset (spot) price.
    strike : float
        Strike price of the option.
    sigma : float
        Volatility of the underlying asset (diffusion component).
    r : float
        Risk-free interest rate.
    T : float
        Time to maturity (in years).
    lambda_ : float
        Jump intensity (average number of jumps per unit time).
    mu_j : float
        Mean of the jump size (log-normal distribution).
    sigma_j : float
        Standard deviation of the jump size (log-normal distribution).
    M : int, optional
        Number of Monte Carlo simulation paths (default is 5000).
    N : int, optional
        Number of time steps in the discretization (default is 5000).

    Returns
    -------
    float
        Estimated option price under the Merton Jump-Diffusion model.
    """

    # Calculating the time-step
    dt = T / N

    # Calculating the jump compensator for merton
    k = np.exp(mu_j + 0.5 * sigma_j * 2) - 1

    # Calculating the drift for the Merton's JD
    drift = (r - lambda_ * k) * dt

    # Calculating the dN_t parameter for the diffusion model.
    dN = np.random.choice(
        [1, 0], size=(M, N), p=lambda_ * dt
    )  # This function assigns the value 1 or 0 based on the probability of lambda * dt

    # Calculating the jump-size based on the definition
    jump_size = np.exp(np.random.normal(mu_j, sigma_j, size=(M, N)))

    # Simulating the asset price dynamics based on the definition of
    # Merton Jump diffusion
    asset_evolution = np.zeros(
        size=(M + 1, N + 1)
    )  # This grid will store the overall evolution of the underlying process

    asset_evolution[:, 0] = (
        spot  # The value of the asset at t=0 will be the spot price.
    )

"""
Inverse Fourier Transform Utility for Carr–Madan Option Pricing.

This module defines the function that computes the inverse Discrete Fourier Transform (IDFT), 
used within the Carr–Madan framework to recover option prices across a grid of strikes 
from their Fourier-transformed representation.

Author: Harsh Parikh  
Date: July 27, 2025
"""

import numpy as np


def psi(alpha, u, char_func, params, t=0.0):
    """
    Computes the Carr–Madan integrand Ψ(u) for Fourier-based option pricing.

    This function evaluates the integrand:
        Ψ(u) = φ(u - i(α + 1)) / (α² + α - u² + i(2α + 1)u),

    where φ is the characteristic function of log(S_T) under the chosen model.
    It is used to compute damped call prices efficiently via inverse Fourier transform.

    Parameters
    ----------
    alpha : float
        Damping parameter α > 0 to ensure square-integrability of the payoff.
    u : complex or np.ndarray
        Frequency domain variable (Fourier space).
    char_func : callable
        The model-specific characteristic function, e.g., `bsm_characteristic_function`.
    params : tuple
        Positional arguments required by `char_func`, typically (r, sigma, spot, T, ...)
    t : float, optional
        Current time (default is 0.0). Only used if not included in `params`.

    Returns
    -------
    complex or np.ndarray
        Value of the Carr–Madan integrand Ψ(u).
    """

    # Fetching the params for the integrand
    r, T = params

    # Shifted frequency for damping
    u_shifted = u - 1j * (alpha + 1)

    # Evaluate the characteristic function at the shifted point
    phi_val = char_func(u_shifted) * np.exp(-r * T)

    # The variable prevents division by zero error
    epsilon = 1e-12

    # Carr–Madan denominator
    denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u + epsilon

    return phi_val / denominator


def carr_madan_fourier_engine(
    spot, strike, sigma, r, T, N, char_func, alpha, dv=0.3, t=0, option_type="call"
):
    """
    Computes the price of a European option using the Carr-Madan FFT method.

    This function implements the Carr–Madan (1999) framework to efficiently compute
    option prices via the Fast Fourier Transform (FFT). The method evaluates the
    damped characteristic function of the log-asset price and recovers option values
    by applying inverse Fourier techniques. This approach is particularly effective
    for pricing across a grid of strikes.

    Input Parameters
    ----------------
    spot: Initial stock price
    strike: Target strike price
    sigma: Volatility of the underlying asset
    r: Risk free rate
    T: Time to maturity
    t: Current time
    dv: Frequency domain spacing
    N: Number of grid points (must be a power of 2)
    alpha: Damping factor
    option_type: Call or put

    Returns
    -------
    The computed option price for the specified strike.
    """

    # Initial check
    if option_type not in ("put", "call"):
        raise ValueError("Please select one from `put` or `call`")

    # Creating grid points for the frequency domain
    freq_grid_pts = dv * np.arange(0, N).astype(np.complex128)

    # Creating log-strike spacing
    dk = 2 * np.pi / (N * dv)  # Nyquist-Shannon condition to avoid aliasing
    b = 0.5 * N * dk  # The `b` term in log-strike spacing

    # Computing the log-strike domain
    log_strike_grid = -b + np.arange(0, N) * dk

    # Computing the `Psi` grid
    psi_grid = np.zeros((N), dtype=np.complex128)

    # Computing the trapezoidal weights
    weights = np.ones((N))
    weights[0] = 0.5  # Weight corresponding to index `0`
    weights[-1] = 0.5  # Weight corresponding to index `N-1`

    # Calculating the Psi grid
    psi_inputs = (
        r,
        T,
    )

    psi_grid = (
        dv
        * psi(alpha, freq_grid_pts, char_func, psi_inputs)
        * np.exp(1j * b * freq_grid_pts)
        * weights
    )

    # Reverting the complex representation of the call prices vector
    call_prices = np.fft.fft(psi_grid)

    # Fetching the strikes grid
    strikes = np.exp(log_strike_grid)

    # Calculating the call price output
    call_price_output = (np.exp(-alpha * log_strike_grid) / np.pi) * call_prices.real

    price = np.interp(strike, strikes, call_price_output)

    return price

"""
Characteristic Function Library for Option Pricing Models.

This module defines characteristic functions for various stochastic models,
such as Black-Scholes, which are compatible with Fourier-based pricing 
techniques like the Carr–Madan framework.

These functions return the characteristic function φ(u) = E[e^{iu log(S_T)}]
under the risk-neutral measure and are designed to be passed directly to 
Fourier-based pricing engines.

Author: Harsh Parikh  
Date: July 27, 2025
"""

import numpy as np


def bsm_characteristic_function(u, r, sigma, spot, T, t=0.0):
    """
    Computes the Black-Scholes characteristic function φ(u) under the risk-neutral measure.

    This function returns the characteristic function φ(u) = E[e^{i u log(S_T)}],
    which is used in Fourier-based pricing frameworks such as Carr–Madan.

    Parameters
    ----------
    u : complex or np.ndarray
        Frequency domain variable (possibly complex).
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying asset.
    spot : float
        Initial asset price.
    T : float
        Time to maturity (in years).
    t : float, optional
        Current time (default is 0.0).

    Returns
    -------
    complex or np.ndarray
        Value of the characteristic function evaluated at u.
    """
    tau = T - t
    phi = np.exp(
        1j * u * (np.log(spot) + (r - 0.5 * sigma**2) * tau)
        - 0.5 * sigma**2 * u**2 * tau
    )
    return phi

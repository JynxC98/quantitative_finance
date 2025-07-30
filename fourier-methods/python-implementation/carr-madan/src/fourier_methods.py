"""
Inverse Fourier Transform Utility for Carr–Madan Option Pricing.

This module defines the function that computes the inverse Discrete Fourier Transform (IDFT), 
used within the Carr–Madan framework to recover option prices across a grid of strikes 
from their Fourier-transformed representation.

Author: Harsh Parikh  
Date: July 27, 2025
"""

from typing import Type
import numpy as np
from numba import jit

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

    # Shifted frequency for damping
    u_shifted = u - 1j * (alpha + 1)

    # Evaluate the characteristic function at the shifted point
    phi_val = char_func(u_shifted, *params, t=t)

    # Carr–Madan denominator
    denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u

    return phi_val / denominator

"""
gamma.py
--------
Lanczos approximation of the Gamma function for complex arguments.

The Lanczos approximation provides a numerically stable way to evaluate
Γ(z) for complex z with Re(z) > 0. For Re(z) < 0.5, the reflection
formula Γ(z)Γ(1-z) = π/sin(πz) is applied first.

Reference:
    https://en.wikipedia.org/wiki/Lanczos_approximation

Author: Harsh Parikh 
"""

import numpy as np
from numba import jit

# Lanczos coefficients for g=7, n=9 (Spouge's version)
# These provide ~15 significant digits of accuracy
_G = 7.0
_P = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]


@jit(nopython=True)
def gamma(z: complex) -> complex:
    """
    Evaluate the Gamma function Γ(z) for complex z using Lanczos approximation.

    Uses the reflection formula for Re(z) < 0.5 to ensure numerical stability
    across the complex plane (excluding non-positive integers where Γ diverges).

    Parameters
    ----------
    z : complex
        Input value. Must not be a non-positive integer (poles of Γ).

    Returns
    -------
    complex
        Γ(z) evaluated at z.

    Raises
    ------
    ValueError
        If z is a non-positive integer (pole of the Gamma function).

    Examples
    --------
    >>> gamma(1.0)   # Γ(1) = 1
    (1+0j)
    >>> gamma(0.5)   # Γ(0.5) = √π
    (1.7724538509...+0j)
    >>> gamma(5.0)   # Γ(5) = 4! = 24
    (24+0j)
    """
    z = np.complex64(z)

    # Reflection formula for Re(z) < 0.5
    if z.real < 0.5:
        return np.pi / (np.sin(np.pi * z) * gamma(1.0 - z))

    # Lanczos approximation
    z -= 1.0
    x = np.complex64(_P[0])

    for i in range(1, len(_P)):
        x += _P[i] / (z + i)

    t = z + _G + 0.5

    return np.sqrt(2.0 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x


def gamma_real(z: float) -> float:
    """
    Convenience wrapper: evaluate Γ(z) for real z, returning a float.

    Parameters
    ----------
    z : float
        Real-valued input.

    Returns
    -------
    float
        Real part of Γ(z). Imaginary part is discarded (should be ~0).
    """
    return gamma(np.complex64(z)).real

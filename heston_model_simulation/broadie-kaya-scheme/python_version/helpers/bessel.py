"""
bessel.py
---------
Modified Bessel function of the first kind I_α(z) for complex arguments.

Two numerical schemes are implemented and selected based on |z|:
  - PowerScheme       : convergent power series, accurate for small |z|
  - AsymptoticExpansion: asymptotic series with minimum-term stopping rule,
                         accurate for large |z|

Both schemes support log-space evaluation to avoid overflow for large |z|,
which is critical for the Broadie-Kaya simulation where Bessel arguments
can be large.

Reference:
    https://en.wikipedia.org/wiki/Bessel_function
    Abramowitz & Stegun, Chapter 9

Author: Harsh Parikh
"""

import numpy as np
from gamma import gamma_real


def _power_scheme(
    z: complex,
    alpha: float,
    tolerance: float = 1e-8,
    num_iterations: int = 100,
    log_space: bool = True,
) -> complex:
    """
    Evaluate I_α(z) via convergent power series (small |z| regime).

    The series is:
        I_α(z) = (z/2)^α * Σ_{k=0}^∞  (z²/4)^k / (k! * Γ(α+k+1))

    The k-th term ratio recurrence is:
        term_k / term_{k-1} = (z²/4) / (k * (α+k))

    In log-space, returns:
        α*log(z/2) + log(Σ term_k / Γ(α+1))

    Parameters
    ----------
    z : complex
        Argument of I_α.
    alpha : float
        Order of the Bessel function.
    tolerance : float
        Convergence tolerance: stop when |term| < tolerance * |sum|.
    num_iterations : int
        Maximum number of series terms.
    log_space : bool
        If True, return log(I_α(z)) to avoid overflow.

    Returns
    -------
    complex
        log(I_α(z)) if log_space=True, else I_α(z).

    Raises
    ------
    RuntimeError
        If the series does not converge within num_iterations.
    """
    # First term k=0: 1 / Γ(α+1)
    term = np.complex64(1.0 / gamma_real(alpha + 1.0))
    total = term

    for k in range(1, num_iterations):
        # Recurrence: term_k = term_{k-1} * (z²/4) / (k*(α+k))
        term *= (0.25 * z * z) / (k * (alpha + k))
        total += term

        if abs(term) < tolerance * abs(total):
            if log_space:
                return alpha * np.log(0.5 * z) + np.log(total)
            else:
                return (0.5 * z) ** alpha * total

    raise RuntimeError(
        f"PowerScheme: convergence not achieved within {num_iterations} iterations "
        f"(|z|={abs(z):.4f}, alpha={alpha:.4f})"
    )


def _asymptotic_expansion(
    z: np.complex64,
    alpha: float,
    tolerance: float = 1e-8,
    num_iterations: int = 100,
    log_space: bool = True,
) -> np.complex64:
    """
    Evaluate I_α(z) via asymptotic expansion (large |z| regime).

    The asymptotic series is:
        I_α(z) ~ e^z / √(2πz) * Σ_{k=0}^∞ (-1)^k * a_k(α) / z^k

    where the term ratio recurrence is:
        term_k / term_{k-1} = -(4α² - (2k-1)²) / (8kz)

    Since this is an asymptotic (divergent) series, the minimum-term
    stopping rule is applied: stop as soon as |term_k| > |term_{k-1}|,
    retaining only the well-behaved partial sum.

    In log-space, returns:
        z - 0.5*log(2πz) + log(Σ terms)

    Parameters
    ----------
    z : np.complex64
        Argument of I_α (should satisfy |z| >> 1 for accuracy).
    alpha : float
        Order of the Bessel function.
    tolerance : float
        Secondary convergence tolerance.
    num_iterations : int
        Maximum number of asymptotic terms.
    log_space : bool
        If True, return log(I_α(z)) to avoid overflow.

    Returns
    -------
    np.complex64
        log(I_α(z)) if log_space=True, else I_α(z).
    """
    term = np.complex64(1.0)
    total = term
    prev_abs = abs(term)

    for k in range(1, num_iterations):
        # term_k/term_{k-1} = -(4α²-(2k-1)²) / (8kz)
        mk = 2.0 * k - 1.0
        term *= -(4.0 * alpha * alpha - mk * mk) / (8.0 * k * z)
        curr_abs = abs(term)

        # Minimum-term stopping rule: divergence detected, stop before adding
        if curr_abs > prev_abs:
            break

        total += term

        if curr_abs < tolerance * abs(total):
            break

        prev_abs = curr_abs

    if log_space:
        return z - 0.5 * np.log(2.0 * np.pi * z) + np.log(total)
    else:
        return (np.exp(z) / np.sqrt(2.0 * np.pi * z)) * total


def modified_bessel(
    z: np.complex64,
    alpha: float,
    num_iterations: int = 100,
    tolerance: float = 1e-10,
    threshold: float = 10.0,
    log_space: bool = True,
) -> np.complex64:
    """
    Evaluate the modified Bessel function of the first kind I_α(z).

    Selects between PowerScheme (|z| ≤ threshold) and AsymptoticExpansion
    (|z| > threshold) automatically. Log-space evaluation is recommended
    for large |z| to avoid floating-point overflow.

    Note
    ----
    Negative integer orders are handled via the symmetry I_{-n}(z) = I_n(z).
    The case alpha=0 with log_space=True raises an error (log(I_0) at z=0
    is undefined; use log_space=False for alpha=0).

    Parameters
    ----------
    z : np.complex64
        Argument of I_α. For Broadie-Kaya, this is a positive real number
        scaled by variance parameters.
    alpha : float
        Order of the Bessel function. In Heston: alpha = d/2 - 1 where
        d = 4κθ/σ² is the degrees of freedom of the CIR process.
    num_iterations : int
        Maximum iterations for either series.
    tolerance : float
        Convergence tolerance.
    threshold : float
        |z| threshold for switching from power series to asymptotic expansion.
    log_space : bool
        If True, return log(I_α(z)). Strongly recommended for large |z|.

    Returns
    -------
    np.complex64
        log(I_α(z)) if log_space=True, else I_α(z).

    Raises
    ------
    ValueError
        If alpha=0 and log_space=True, or if z=0 with alpha < 0.

    Examples
    --------
    >>> # Compare with scipy for validation
    >>> from scipy.special import iv
    >>> z, alpha = 5.0, 1.0
    >>> result = modified_bessel(z, alpha, log_space=False).real
    >>> expected = iv(alpha, z)
    >>> abs(result - expected) < 1e-6
    True
    """
    z = np.complex64(z)

    if log_space and alpha == 0:
        raise ValueError("log_space=True is undefined at alpha=0 when z=0.")

    # Symmetry: I_{-n}(z) = I_n(z) for integer n
    if alpha < 0 and abs(round(alpha) - alpha) < 1e-12:
        return modified_bessel(
            z, -alpha, num_iterations, tolerance, threshold, log_space
        )

    # Base cases at z=0
    if z == 0.0:
        if alpha < 0:
            raise ValueError("I_α(0) diverges for α < 0")
        # I_0(0) = 1 → log = 0; I_α(0) = 0 for α > 0 → log = -inf
        return np.complex64(0.0)

    if abs(z) <= threshold:
        return _power_scheme(z, alpha, tolerance, num_iterations, log_space)
    else:
        return _asymptotic_expansion(z, alpha, tolerance, num_iterations, log_space)

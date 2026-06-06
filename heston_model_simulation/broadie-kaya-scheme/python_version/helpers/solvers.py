"""
solvers.py
----------
Numerical inversion of the integrated variance CDF for Broadie-Kaya
exact simulation of the Heston stochastic volatility model.

The CDF is computed via the Gil-Pelaez Fourier inversion formula:

    F(x) = ½ - (1/π) ∫₀^∞  Im(e^{-iux} φ(u)) / u  du

where φ(u) is the conditional characteristic function of ∫_t^{t+dt} V_s ds
given V_t and V_{t+dt}.

The PDF and its derivative (needed for Halley's method) follow by
differentiating under the integral sign:

    f(x)  = (1/π) ∫₀^∞  Re(e^{-iux} φ(u))      du
    f'(x) = (1/π) ∫₀^∞  u · Im(e^{-iux} φ(u))  du

Damping
-------
All three integrands use the same Laplace-damped characteristic function

    φ_δ(u) = φ(u + iδ),   δ = damp = 0.005

and the matching exponential prefactor exp(-δ u) so that differentiation
under the integral sign remains consistent:

    CDF integrand  : -exp(-δu) · Im[e^{-iux} φ_δ(u)] / (uπ)
    PDF integrand  :  exp(-δu) · Re[e^{-iux} φ_δ(u)] / π
    dPDF integrand :  exp(-δu) · u · Im[e^{-iux} φ_δ(u)] / π

Note: the PDF integrand is the exact derivative of the CDF integrand with
respect to x, and dPDF is the exact derivative of the PDF integrand.
All three must carry the same exp(-δu) prefactor for this consistency.

Quadrature strategy
-------------------
Breakpoints are scaled by 1/max(|x|, 1) so that each sub-interval
contains O(1) oscillation cycles regardless of x.  The number of
sub-intervals is capped at 500 to prevent runaway cost for very small x.

Solvers
-------
Primary   : Halley's method (cubic convergence near the root).
Fallback  : Brent's method via scipy.optimize.brentq (guaranteed convergence).
            Triggered automatically if Halley diverges, the denominator
            becomes dangerously small, or max iterations is exceeded.

JIT note
--------
The pure-arithmetic Halley step (_halley_step) is decorated with
@numba.njit for a modest speed-up.  The bottleneck is the three quad
calls per iteration; for large-scale simulation parallelize across
independent quantile inversions using multiprocessing.Pool or joblib.

Reference
---------
Broadie, M. & Kaya, O. (2006). Exact simulation of stochastic
volatility and other affine jump diffusion processes.
Operations Research, 54(2), 217-244.

Gil-Pelaez, J. (1951). Note on the inversion theorem.
Biometrika, 38(3-4), 481-482.

Author: Harsh Parikh
"""

import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from numba import njit

from char_func import HestonParams, char_function

# ---------------------------------------------------------------------------
# Damping constant
# ---------------------------------------------------------------------------
# A small positive shift δ applied to the characteristic function argument:
#   φ_δ(u) = φ(u + iδ)
# This regularises the u→0 singularity in the CDF integrand and improves
# quadrature convergence.  δ must be strictly less than the distance from
# the real axis to the nearest pole of φ in the upper half-plane.
# For typical Heston parameters, δ = 0.005 is safe; adjust if φ blows up.
damp: float = 0.005

# ---------------------------------------------------------------------------
# Internal: Halley arithmetic step (JIT-compiled)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _halley_step(
    x: float,
    f: float,
    f_prime: float,
    f_double_prime: float,
) -> tuple:
    """
    Compute one Halley update.

    Halley's rule:

        x_{n+1} = x_n - (2 f f') / (2(f')² - f f'')

    Parameters
    ----------
    x : float
        Current iterate.
    f : float
        F(x) - var  (residual).
    f_prime : float
        f(x)  (PDF at x).
    f_double_prime : float
        f'(x) (derivative of PDF at x).

    Returns
    -------
    x_new : float
        Updated iterate (reflected to positive half-line if needed).
    ok : bool
        False if the denominator was dangerously small (triggers fallback).
    """
    denom = 2.0 * f_prime**2 - f * f_double_prime

    if abs(denom) < 1e-14:
        return x, False  # signal fallback to caller

    update = (2.0 * f * f_prime) / denom
    x_new = x - update

    # Integrated variance is strictly positive
    if x_new <= 0.0:
        x_new = x / 2.0

    return x_new, True


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------


def _calculate_integral(integrand_fn, x: float, p: HestonParams) -> float:
    """
    Integrate an oscillatory CF-inversion integrand from 0 to 500.

    Breakpoints are scaled by 1/max(|x|, 1) so that each sub-interval
    spans roughly one oscillation period (2π/x), keeping quadrature error
    bounded as x grows.  The number of breakpoints is capped at 500.

    Parameters
    ----------
    integrand_fn : callable
        One of cdf_integrand, pdf_integrand, d_pdf_integrand.
    x : float
        Current quantile estimate (controls oscillation frequency).
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        Numerical value of the integral.
    """
    upper = 500.0

    # Number of sub-intervals ~ number of oscillation cycles in [0, upper]
    n_intervals = max(50, int(upper * abs(x) / (2.0 * np.pi)) + 1)
    n_intervals = min(n_intervals, 500)  # cap to prevent runaway cost

    break_points = np.linspace(0.0, upper, n_intervals + 1)

    result = 0.0
    for i in range(len(break_points) - 1):
        val, _ = quad(
            lambda u: integrand_fn(x, u, p),
            a=break_points[i],
            b=break_points[i + 1],
            limit=100,
            epsabs=1e-8,
            epsrel=1e-8,
        )
        result += val

    return result


# ---------------------------------------------------------------------------
# Integrands
# ---------------------------------------------------------------------------


def cdf_integrand(x: float, u: float, p: HestonParams) -> float:
    """
    Integrand of the Gil-Pelaez CDF inversion formula (with damping).

        g(u) = -exp(-δu) · Im[e^{-iux} φ_δ(u)] / (u π)

    where φ_δ(u) = φ(u + iδ) is the damped characteristic function.

    The product e^{-iux} φ_δ(u) expands as:

        Im[e^{-iux} φ_δ] = cos(ux) · Im(φ_δ) - sin(ux) · Re(φ_δ)

    Handles the u→0 singularity analytically:

        lim_{u→0} g(u) = x/π

    (since φ(0) = 1, e^{-iux} ≈ 1 - iux as u → 0).

    Parameters
    ----------
    x : float
        Quantile at which the CDF is evaluated.
    u : float
        Integration variable (frequency).
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        Integrand value at u.
    """
    if abs(u) < 1e-8:
        # Analytic limit removes the 1/u singularity
        return x / np.pi

    phi = char_function(p, u + 1j * damp)

    # Im[e^{-iux} φ_δ(u)] = cos(ux)·Im(φ_δ) - sin(ux)·Re(φ_δ)
    im_part = np.cos(u * x) * phi.imag - np.sin(u * x) * phi.real

    return -np.exp(-damp * u) * im_part / (u * np.pi)


def pdf_integrand(x: float, u: float, p: HestonParams) -> float:
    """
    Integrand of the Fourier inversion formula for the PDF (with damping).

        g(u) = exp(-δu) · Re[e^{-iux} φ_δ(u)] / π

    This is the exact derivative of cdf_integrand with respect to x,
    which is why the same exp(-δu) prefactor is required for consistency.

    The product e^{-iux} φ_δ(u) expands as:

        Re[e^{-iux} φ_δ] = cos(ux) · Re(φ_δ) + sin(ux) · Im(φ_δ)

    Handles the u→0 limit analytically (→ 1/π, since φ(0) = 1).

    Parameters
    ----------
    x : float
        Quantile at which the PDF is evaluated.
    u : float
        Integration variable (frequency).
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        Integrand value at u.
    """
    if abs(u) < 1e-8:
        return 1.0 / np.pi  # lim_{u→0} Re[e^{-iux} φ(u)] / π = 1/π

    phi = char_function(p, u + 1j * damp)

    # Re[e^{-iux} φ_δ(u)] = cos(ux)·Re(φ_δ) + sin(ux)·Im(φ_δ)
    re_part = np.cos(u * x) * phi.real + np.sin(u * x) * phi.imag

    return np.exp(-damp * u) * re_part / np.pi


def d_pdf_integrand(x: float, u: float, p: HestonParams) -> float:
    """
    Integrand for f'(x), the first derivative of the PDF (with damping).

        g(u) = exp(-δu) · u · Im[e^{-iux} φ_δ(u)] / π

    This is the exact derivative of pdf_integrand with respect to x.
    The same exp(-δu) prefactor is required for differentiation under
    the integral sign to hold.

    Parameters
    ----------
    x : float
        Quantile at which f'(x) is evaluated.
    u : float
        Integration variable (frequency).
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        Integrand value at u.
    """
    if abs(u) < 1e-8:
        return 0.0  # u · Im[...] / π → 0 as u → 0

    phi = char_function(p, u + 1j * damp)

    # Im[e^{-iux} φ_δ(u)] = cos(ux)·Im(φ_δ) - sin(ux)·Re(φ_δ)
    im_part = np.cos(u * x) * phi.imag - np.sin(u * x) * phi.real

    return np.exp(-damp * u) * (u / np.pi) * im_part


# ---------------------------------------------------------------------------
# Public CDF / PDF interface
# ---------------------------------------------------------------------------


def calculate_cdf(x: float, p: HestonParams) -> float:
    """
    Evaluate F(x) = P(∫_t^{t+dt} V_s ds ≤ x) via Gil-Pelaez inversion.

        F(x) = ½ + ∫₀^∞ cdf_integrand(x, u, p) du

    Parameters
    ----------
    x : float
        Quantile value (must be > 0 for the variance integral).
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        CDF value in [0, 1].
    """
    return 0.5 + _calculate_integral(cdf_integrand, x, p)


def calculate_pdf(x: float, p: HestonParams) -> float:
    """
    Evaluate f(x), the PDF of ∫_t^{t+dt} V_s ds, via Fourier inversion.

        f(x) = ∫₀^∞ pdf_integrand(x, u, p) du

    Parameters
    ----------
    x : float
        Quantile value.
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        PDF value (non-negative).
    """
    return _calculate_integral(pdf_integrand, x, p)


def calculate_d_pdf(x: float, p: HestonParams) -> float:
    """
    Evaluate f'(x), the derivative of the PDF, via Fourier inversion.

        f'(x) = ∫₀^∞ d_pdf_integrand(x, u, p) du

    Parameters
    ----------
    x : float
        Quantile value.
    p : HestonParams
        Heston model parameters.

    Returns
    -------
    float
        f'(x) value.
    """
    return _calculate_integral(d_pdf_integrand, x, p)


# ---------------------------------------------------------------------------
# Initial guess
# ---------------------------------------------------------------------------


def _initial_guess(var: float, p: HestonParams) -> float:
    """
    Compute a starting point for the root-finder.

    Uses the analytical mean of the CIR-integrated variance:

        E[∫_0^dt V_s ds | V_0] = θ·dt + (V_0 - θ)·(1 - e^{-κ·dt}) / κ

    which degenerates to V_0·dt when κ → 0.  The mean is then shifted
    toward the appropriate tail based on the target probability var.

    Parameters
    ----------
    var : float
        Target probability in (0, 1).
    p : HestonParams
        Heston model parameters (kappa, theta, dt, v_t used).

    Returns
    -------
    float
        Initial guess x_0 > 0.
    """
    kappa, theta, dt, v0 = p.kappa, p.theta, p.dt, p.v_t

    if abs(kappa) < 1e-10:
        mean = v0 * dt
    else:
        mean = theta * dt + (v0 - theta) * (1.0 - np.exp(-kappa * dt)) / kappa

    # Shift toward lower/upper tail proportionally
    return max(mean * (0.1 + var * 1.8), 1e-6)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def run_halley_solver(
    var: float,
    p: HestonParams,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Invert F(x) = var using Halley's (Newton's second-order) method.

    Halley's update rule gives cubic convergence near the root:

        x_{n+1} = x_n - (2 f f') / (2(f')² - f f'')

    where f = F(x) - var, f' = PDF(x), f'' = dPDF(x).

    Falls back to Brent's method automatically if:
      - the Halley denominator 2(f')² - f·f'' is near zero, or
      - the iterate has not converged within max_iterations steps.

    Parameters
    ----------
    var : float
        Target probability in (0, 1).
    p : HestonParams
        Heston model parameters.
    tolerance : float
        Convergence criterion: |x_{n+1} - x_n| < tolerance.
    max_iterations : int
        Maximum Halley iterations before falling back to Brent.

    Returns
    -------
    float
        Quantile x* such that F(x*) ≈ var.

    Raises
    ------
    ValueError
        If var is not in (0, 1).
    RuntimeError
        If both Halley and Brent solvers fail to converge.
    """
    if not (0.0 < var < 1.0):
        raise ValueError(f"var must be in (0, 1), got {var}")

    x = _initial_guess(var, p)

    for i in range(max_iterations):
        f = calculate_cdf(x, p) - var
        f_prime = calculate_pdf(x, p)
        f_double_prime = calculate_d_pdf(x, p)

        x_new, ok = _halley_step(x, f, f_prime, f_double_prime)

        if not ok:
            warnings.warn(
                f"Halley denominator near zero at iteration {i} (x={x:.6f}). "
                "Falling back to Brent's method.",
                RuntimeWarning,
                stacklevel=2,
            )
            return run_brent_solver(var, p, tolerance)

        if abs(x_new - x) < tolerance:
            return x_new

        x = x_new

    warnings.warn(
        f"Halley solver did not converge in {max_iterations} iterations. "
        "Falling back to Brent's method.",
        RuntimeWarning,
        stacklevel=2,
    )
    return run_brent_solver(var, p, tolerance)


def run_brent_solver(
    var: float,
    p: HestonParams,
    tolerance: float = 1e-8,
) -> float:
    """
    Invert F(x) = var using Brent's method (guaranteed convergence).

    Brent's method combines bisection, secant, and inverse quadratic
    interpolation.  It is used as:
      - A standalone reference solver for validating Halley results.
      - A guaranteed fallback when Halley diverges.

    The bracketing interval [lo, hi] is grown adaptively until a sign
    change is detected, making it robust across parameter regimes.

    Parameters
    ----------
    var : float
        Target probability in (0, 1).
    p : HestonParams
        Heston model parameters.
    tolerance : float
        Absolute tolerance passed to scipy.optimize.brentq.

    Returns
    -------
    float
        Quantile x* such that F(x*) ≈ var.

    Raises
    ------
    ValueError
        If var is not in (0, 1).
    RuntimeError
        If a bracketing interval cannot be found within 50 expansions.
    """
    if not (0.0 < var < 1.0):
        raise ValueError(f"var must be in (0, 1), got {var}")

    def objective(x: float) -> float:
        return calculate_cdf(x, p) - var

    # Build bracket adaptively from the CIR mean
    kappa, theta, dt, v0 = p.kappa, p.theta, p.dt, p.v_t
    if abs(kappa) < 1e-10:
        mean = v0 * dt
    else:
        mean = theta * dt + (v0 - theta) * (1.0 - np.exp(-kappa * dt)) / kappa

    lo = max(mean * 1e-3, 1e-9)
    hi = mean * 10.0

    for _ in range(50):
        if objective(lo) * objective(hi) < 0.0:
            break
        lo /= 2.0
        hi *= 2.0
    else:
        raise RuntimeError(
            f"Could not bracket root for var={var:.4f}. " f"Check parameters: {p}"
        )

    return brentq(objective, lo, hi, xtol=tolerance, maxiter=200)

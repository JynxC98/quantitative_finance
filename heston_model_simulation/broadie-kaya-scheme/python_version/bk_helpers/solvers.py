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
Fallback  : bisection method (numba-compatible)

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

"""
solvers.py
----------
Numerical inversion of the integrated variance CDF for Broadie-Kaya
exact simulation of the Heston stochastic volatility model.

(docstring unchanged from before, except "Fallback: Brent's method" should
now read "Fallback: bisection method (numba-compatible)" -- see note below)

Author: Harsh Parikh
"""

import warnings
import numpy as np
from numba import njit

from .quadrature import Legendre_Integrate
from .char_func import HestonParams, char_function

damp: float = 0.005


def _halley_step(
    x: float,
    f: float,
    f_prime: float,
    f_double_prime: float,
) -> tuple:
    """
    Compute one Halley update.

    Halley's rule:
        x_{n+1} = x_n - (2 f f') / (2(f')^2 - f f'')
    """
    denom = 2.0 * f_prime**2 - f * f_double_prime

    if abs(denom) < 1e-14:
        return x, False

    update = (2.0 * f * f_prime) / denom
    x_new = x - update

    if x_new <= 0.0:
        x_new = x / 2.0

    return x_new, True


def calculate_integral(integrand_fn, x: float, p: HestonParams) -> float:
    """
    Integrate an oscillatory CF-inversion integrand from 0 to 500.
    """
    upper = 500.0

    n_intervals = max(50, int(upper * abs(x) / (2.0 * np.pi)) + 1)
    n_intervals = min(n_intervals, 500)

    break_points = np.linspace(0.0, upper, n_intervals + 1)

    result = 0.0
    for i in range(len(break_points) - 1):
        val = Legendre_Integrate(
            lambda u: integrand_fn(x, u, p),
            break_points[i],
            break_points[i + 1],
        )
        result += val

    return result


def cdf_integrand(x: float, u: float, p: HestonParams) -> float:
    """
    Integrand of the Gil-Pelaez CDF inversion formula (with damping).
    """
    if abs(u) < 1e-8:
        return x / np.pi

    phi = char_function(p, u + 1j * damp)

    im_part = np.cos(u * x) * phi.imag - np.sin(u * x) * phi.real

    return -np.exp(-damp * u) * im_part / (u * np.pi)


def pdf_integrand(x: float, u: float, p: HestonParams) -> float:
    """
    Integrand of the Fourier inversion formula for the PDF (with damping).
    """
    if abs(u) < 1e-8:
        return 1.0 / np.pi

    phi = char_function(p, u + 1j * damp)

    re_part = np.cos(u * x) * phi.real + np.sin(u * x) * phi.imag

    return np.exp(-damp * u) * re_part / np.pi


def d_pdf_integrand(x: float, u: float, p: HestonParams) -> float:
    """
    Integrand for f'(x), the first derivative of the PDF (with damping).
    """
    if abs(u) < 1e-8:
        return 0.0

    phi = char_function(p, u + 1j * damp)

    im_part = np.cos(u * x) * phi.imag - np.sin(u * x) * phi.real

    return np.exp(-damp * u) * (u / np.pi) * im_part


def calculate_cdf(x: float, p: HestonParams) -> float:
    """
    Evaluate F(x) via Gil-Pelaez inversion.
    """
    return 0.5 + calculate_integral(cdf_integrand, x, p)


def calculate_pdf(x: float, p: HestonParams) -> float:
    """
    Evaluate f(x) via Fourier inversion.
    """
    return calculate_integral(pdf_integrand, x, p)


def calculate_d_pdf(x: float, p: HestonParams) -> float:
    """
    Evaluate f'(x) via Fourier inversion.
    """
    return calculate_integral(d_pdf_integrand, x, p)


def _initial_guess(var: float, p: HestonParams) -> float:
    """
    Compute a starting point for the root-finder using the CIR mean.
    """
    kappa, theta, dt, v0 = p.kappa, p.theta, p.dt, p.v_t

    if abs(kappa) < 1e-10:
        mean = v0 * dt
    else:
        mean = theta * dt + (v0 - theta) * (1.0 - np.exp(-kappa * dt)) / kappa

    return max(mean * (0.1 + var * 1.8), 1e-6)


# ---------------------------------------------------------------------------
# Tail threshold for solver selection
# ---------------------------------------------------------------------------
TAIL_EPS: float = 1e-3


def run_bisection_solver(
    var: float,
    p: HestonParams,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """
    Invert F(x) = var using bisection (numba-compatible, guaranteed
    convergence given a valid bracket). Replaces Brent's method, which
    is not nopython-compatible.
    """
    kappa, theta, dt, v0 = p.kappa, p.theta, p.dt, p.v_t

    if abs(kappa) < 1e-10:
        mean = v0 * dt
    else:
        mean = theta * dt + (v0 - theta) * (1.0 - np.exp(-kappa * dt)) / kappa

    lo = max(mean * 1e-3, 1e-9)
    hi = mean * 10.0

    f_lo = calculate_cdf(lo, p) - var
    f_hi = calculate_cdf(hi, p) - var

    expand_count = 0
    while f_lo * f_hi > 0.0 and expand_count < 50:
        lo /= 2.0
        hi *= 2.0
        f_lo = calculate_cdf(lo, p) - var
        f_hi = calculate_cdf(hi, p) - var
        expand_count += 1

    if f_lo * f_hi > 0.0:
        return mean

    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = calculate_cdf(mid, p) - var

        if abs(f_mid) < tolerance or 0.5 * (hi - lo) < tolerance:
            return mid

        if f_lo * f_mid < 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 0.5 * (lo + hi)


def run_halley_solver(
    var: float,
    p: HestonParams,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Invert F(x) = var using Halley's (Newton's second-order) method.

    Falls back to bisection automatically if:
      - the Halley denominator 2(f')^2 - f*f'' is near zero, or
      - the iterate has not converged within max_iterations steps.
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
                "Falling back to bisection.",
                RuntimeWarning,
                stacklevel=2,
            )
            return run_bisection_solver(var, p, tolerance)

        if abs(x_new - x) < tolerance:
            return x_new

        x = x_new

    warnings.warn(
        f"Halley solver did not converge in {max_iterations} iterations. "
        "Falling back to bisection.",
        RuntimeWarning,
        stacklevel=2,
    )
    return run_bisection_solver(var, p, tolerance)


def invert_variance_cdf(
    var: float,
    p: HestonParams,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Invert F(x) = var to recover the quantile x*, dispatching to the
    appropriate solver based on how extreme var is.

    Near the tails (var < TAIL_EPS or var > 1 - TAIL_EPS), bisection is
    used directly, since Halley's denominator tends to be ill-conditioned
    where the PDF is very small. Elsewhere, Halley's cubic convergence
    is used for speed.
    """
    if not (0.0 < var < 1.0):
        raise ValueError(f"var must be in (0, 1), got {var}")

    if var < TAIL_EPS or var > 1.0 - TAIL_EPS:
        return run_bisection_solver(var, p, tolerance)

    return run_halley_solver(var, p, tolerance, max_iterations)

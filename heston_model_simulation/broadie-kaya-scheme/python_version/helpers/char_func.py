"""
char_function.py
----------------
Characteristic function of the conditional integrated variance process
in the Heston stochastic volatility model, as required for Broadie-Kaya
exact simulation.

The characteristic function φ(u) = E[e^{iuV_t} | V_0] involves:
  1. A scaling term (first_term)
  2. An exponential moment term (second_term)
  3. A ratio of modified Bessel functions (third_term, evaluated in log-space)

Parameters are encapsulated in HestonParams. All intermediate quantities
are computed in log-space where possible to avoid overflow.

Reference:
    Broadie, M. & Kaya, O. (2006). Exact simulation of stochastic volatility
    and other affine jump diffusion processes. Operations Research, 54(2).

Author: Harsh Parikh
"""

import numpy as np
from dataclasses import dataclass
from bessel import modified_bessel


@dataclass
class HestonParams:
    """
    Container for Heston model parameters used in Broadie-Kaya simulation.

    Attributes
    ----------
    kappa : float
        Mean reversion speed of the variance process (κ > 0).
    theta : float
        Long-run variance (θ > 0).
    sigma : float
        Volatility of variance / vol-of-vol (σ > 0).
    v_t : float
        Current variance V_t (initial condition, v_t > 0).
    v_u : float
        Target variance V_u at time u (v_u > 0).
    dt : float
        Time step Δt = u - t (dt > 0).

    Notes
    -----
    The Feller condition 2κθ > σ² ensures the variance process stays
    strictly positive. While the simulation works without it, satisfying
    Feller avoids degenerate behaviour near zero.
    """

    kappa: float  # Mean reversion speed
    theta: float  # Long-run variance
    sigma: float  # Vol of vol
    v_t: float  # Variance at time t (start)
    v_u: float  # Variance at time u (end)
    dt: float  # Time step


def char_function(p: HestonParams, u: float) -> complex:
    """
    Evaluate the conditional characteristic function of the integrated
    variance ∫_t^u V_s ds given V_t and V_u.

    The characteristic function takes the form:
        φ(u) = first_term * second_term * third_term

    where:
        first_term  = [γ * exp(-½(γ-κ)Δt) * (1-exp(-κΔt))] /
                      [κ * (1-exp(-γΔt))]

        second_term = exp([(v_u+v_t)/σ²] *
                      [κ(1+exp(-κΔt))/(1-exp(-κΔt)) -
                       γ(1+exp(-γΔt))/(1-exp(-γΔt))])

        third_term  = I_α(z_num) / I_α(z_den)   (evaluated in log-space)

    with:
        γ = √(κ² - 2σ²iu)
        α = d/2 - 1,  d = 4κθ/σ²
        z_num, z_den = scaled Bessel arguments

    Parameters
    ----------
    p : HestonParams
        Heston model parameters.
    u : float
        Frequency argument of the characteristic function.

    Returns
    -------
    complex
        φ(u): the characteristic function evaluated at u.
        φ(0) = 1 by convention (handled explicitly).

    Examples
    --------
    >>> p = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, v_t=0.04, v_u=0.04, dt=1/12)
    >>> cf = char_function(p, 0.0)
    >>> abs(cf - 1.0) < 1e-10  # φ(0) = 1
    True
    >>> abs(char_function(p, 1.0)) <= 1.0  # |φ(u)| ≤ 1
    True
    """
    i = complex(0.0, 1.0)

    # φ(0) = 1 by definition (guard against 0/0)
    if abs(u) < 1e-12:
        return complex(1.0, 0.0)

    # γ = √(κ² - 2σ²iu) — the fundamental frequency of the CF
    const_gamma = np.sqrt(p.kappa**2 - 2.0 * p.sigma**2 * i * u)

    # Precompute exponentials shared across terms
    exp_kappa_dt = np.exp(-p.kappa * p.dt)
    exp_gamma_dt = np.exp(-const_gamma * p.dt)

    # --- First term ---
    # Scaling factor relating CIR transition under u-tilted measure to original
    first_term = (
        const_gamma
        * np.exp(-0.5 * (const_gamma - p.kappa) * p.dt)
        * (1.0 - exp_kappa_dt)
    ) / (p.kappa * (1.0 - exp_gamma_dt))

    # --- Second term ---
    # Exponential moment of (v_t + v_u) weighted by cotangent-like expressions
    coth_kappa = (1.0 + exp_kappa_dt) / (1.0 - exp_kappa_dt)
    coth_gamma = (1.0 + exp_gamma_dt) / (1.0 - exp_gamma_dt)

    second_term = np.exp(
        ((p.v_u + p.v_t) / p.sigma**2)
        * (p.kappa * coth_kappa - const_gamma * coth_gamma)
    )

    # --- Third term: ratio of modified Bessel functions in log-space ---
    d = 4.0 * p.kappa * p.theta / p.sigma**2  # degrees of freedom
    alpha = 0.5 * d - 1.0  # Bessel order

    # Bessel argument under the u-tilted measure (numerator)
    bessel_arg_num = np.sqrt(p.v_u * p.v_t) * (
        (4.0 * const_gamma * np.exp(-0.5 * const_gamma * p.dt))
        / (p.sigma**2 * (1.0 - exp_gamma_dt))
    )

    # Bessel argument under the original measure (denominator)
    bessel_arg_den = np.sqrt(p.v_u * p.v_t) * (
        (4.0 * p.kappa * np.exp(-0.5 * p.kappa * p.dt))
        / (p.sigma**2 * (1.0 - exp_kappa_dt))
    )

    log_bessel_num = modified_bessel(bessel_arg_num, alpha, log_space=True)
    log_bessel_den = modified_bessel(bessel_arg_den, alpha, log_space=True)

    # Ratio in log-space → exp(log_num - log_den)
    third_term = np.exp(log_bessel_num - log_bessel_den)

    return first_term * second_term * third_term

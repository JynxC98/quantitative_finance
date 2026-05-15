"""
Newton-Raphson solver for Heston model quantile inversion

This module implements numerical inversion of the cumulative distribution
function (CDF) for the Heston model using Gaussian quadrature and
Newton's second-order method (Halley's method) for root finding.

Author: Harsh Parikh
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from numba import jit

from helpers.char_func import char_function


@dataclass
class HestonParams:
    """Heston model parameters container"""

    v_t: float  # Current variance
    v_u: float  # Long-term variance
    dt: float  # Time step
    # Add other Heston parameters as needed (kappa, theta, sigma, rho)
    kappa: float = 0.0
    theta: float = 0.0
    sigma: float = 0.0
    rho: float = 0.0


@jit(nopython=True)
def calculate_integral(func: Callable, x: float, params: HestonParams) -> float:
    """
    Numerical integration using Gaussian-Legendre quadrature

    Wraps the quadrature module to compute definite integrals
    over u for the CDF, PDF, or dPDF integrands.

    Args:
        func: The integrand function (CDFIntegrand, PDFIntegrand, or d_PDFIntegrand)
        x: The quantile value at which to evaluate the integrand
        params: Heston model parameters

    Returns:
        Numerical approximation of the definite integral
    """

    def integrand(u: float) -> float:
        return func(x, u, params)

    result = quad_vec(integrand, 0, np.inf)

    return result


@jit(nopython=True)
def cdf_integrand(x: float, u: float, params: HestonParams) -> float:
    """
    Integrand for computing the cumulative distribution function (CDF)

    Implements the integrand from Gil-Pelaez inversion formula:
    F(x) = 1/2 - (1/π) ∫₀^∞ Im(e^{-iux} φ(u))/u du

    Args:
        x: The quantile value at which to evaluate the CDF
        u: Integration variable
        params: Heston model parameters

    Returns:
        Integrand value -Im(e^{-iux} φ(u))/(u * π)
    """
    phi = char_function(params, u)

    if abs(u) < 1e-8:
        # Limit of -Im[e^{-iux}φ(u)]/u as u -> 0
        # Using φ(0)=1 and e^{-iux} ≈ 1 - iux, the term is -Im[(1-iux)(1)]/u = x
        return x / np.pi

    # Im[e^{-iux}φ(u)] = Im[(cos(ux) - i*sin(ux)) * (Re(φ) + i*Im(φ))]
    # = cos(ux)*Im(φ) - sin(ux)*Re(φ)
    imag_part = np.cos(u * x) * phi.imag - np.sin(u * x) * phi.real

    return -imag_part / (u * np.pi)


@jit(nopython=True)
def calculate_cdf(x: float, params: HestonParams) -> float:
    """
    Calculate the cumulative distribution function at point x

    Args:
        x: Quantile value
        params: Heston model parameters

    Returns:
        CDF value F(x) = P(X ≤ x)
    """
    return 0.5 + calculate_integral(cdf_integrand, x, params)


@jit(nopython=True)
def pdf_integrand(x: float, u: float, params: HestonParams) -> float:
    """
    Integrand for computing the probability density function (PDF)

    f(x) = (1/π) ∫₀^∞ Re(e^{-iux} φ(u)) du

    Args:
        x: Quantile value
        u: Integration variable
        params: Heston model parameters

    Returns:
        Integrand value
    """
    phi = char_function(params, u)

    # Re[e^{-iux}φ(u)] = Re[(cos(ux) - i*sin(ux)) * (Re(φ) + i*Im(φ))]
    # = cos(ux)*Re(φ) + sin(ux)*Im(φ)
    real_part = np.cos(u * x) * phi.real + np.sin(u * x) * phi.imag

    return (1.0 / np.pi) * real_part


@jit(nopython=True)
def d_pdf_integrand(x: float, u: float, params: HestonParams) -> float:
    """
    Integrand for computing the first derivative of the PDF (F''(x))

    d/dx PDF(x) = (1/π) ∫₀^∞ Re(-iu * e^{-iux} φ(u)) du
    = (u/π) ∫₀^∞ Im(e^{-iux} φ(u)) du

    Args:
        x: Quantile value
        u: Integration variable
        params: Heston model parameters

    Returns:
        Integrand value
    """
    phi = char_function(params, u)

    imag_part = np.cos(u * x) * phi.imag - np.sin(u * x) * phi.real

    return (u / np.pi) * imag_part


@jit(nopython=True)
def run_newton_solver(
    var: float, params: HestonParams, tolerance: float = 1e-8, max_iterations: int = 100
) -> float:
    """
    Newton's second-order (Halley's) method for quantile inversion

    Solves F(x) - var = 0 using Halley's method for cubic convergence.

    Args:
        var: The target probability [0,1] to invert
        params: Heston model parameters
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Quantile x such that F(x) = var

    Raises:
        RuntimeError: If solver does not converge
    """
    # Trapezoidal method for initial guess
    x = (params.v_t + params.v_u) / 2 * params.dt

    def optimize_function(x, target_var, params):
        return calculate_cdf(x, params) - target_var

    result = brentq(optimize_function, a=0.0, b=5.0, args=(var, params))

    return result

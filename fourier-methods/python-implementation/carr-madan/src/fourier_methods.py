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


class CharacteristicFunction:
    """
    Base class for characteristic functions of stochastic models used in
    Fourier-based option pricing frameworks such as Carr–Madan.

    This class provides a common structure to store basic option and market parameters
    like the risk-free rate, volatility, spot price, time to maturity, and frequency domain
    variable `u`. Model-specific characteristic functions should inherit from this base class.

    Parameters
    ----------
    u : complex or np.ndarray
        Frequency domain variable (may be a single complex value or an array).
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying asset.
    spot : float
        Initial spot price of the underlying asset.
    T : float
        Time to maturity (in years).
    t : float, optional
        Current time (default is 0.0).
    """

    def __init__(self, u, r, sigma, spot, T, t=0.0):
        self.u = u
        self.r = r
        self.sigma = sigma
        self.spot = spot
        self.tau = T - t


class BSMCharacteristicFunction(CharacteristicFunction):
    """
    Characteristic function φ(u) under the risk-neutral Black-Scholes model.

    This is a critical component in Fourier-based pricing techniques like Carr–Madan,
    where φ is evaluated at complex-shifted arguments to ensure square-integrability
    of the damped call price.

    Parameters
    ----------
    u : complex or np.ndarray
        Frequency domain variable (possibly complex).
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    spot : float
        Spot price of the underlying.
    T : float
        Time to maturity.
    t : float, optional
        Current time (default is 0.0).
    """

    def __init__(self, u, r, sigma, spot, T, t=0.0):
        super().__init__(u=u, r=r, sigma=sigma, spot=spot, T=T, t=t)

    def calculate_phi(self):
        """
        Computes the Black-Scholes characteristic function φ(u).

        Returns
        -------
        complex or np.ndarray
            Value of the characteristic function evaluated at `u`.
        """
        phi = np.exp(
            1j
            * self.u
            * (np.log(self.spot) + (self.r - 0.5 * self.sigma**2) * self.tau)
            - 0.5 * self.sigma**2 * self.u**2 * self.tau
        )
        return phi


class HestonCharacteristicFunction(CharacteristicFunction):
    """
    Characteristic function φ(u) under the Heston stochastic volatility model.

    The Heston model assumes stochastic variance following a CIR process:
        dS_t = r S_t dt + sqrt(v_t) S_t dW_1(t)
        dv_t = κ(θ - v_t) dt + σ√v_t dW_2(t)
    with correlation ρ between dW₁ and dW₂.

    The characteristic function is defined as:
        φ(u) = E[e^{iu log(S_T)}]
    and has a semi-analytical closed form useful in Fourier-based methods like Carr–Madan.

    Parameters
    ----------
    u : complex or np.ndarray
        Frequency domain variable.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of volatility (σ).
    spot : float
        Initial asset price.
    T : float
        Time to maturity.
    t : float, optional
        Current time (default is 0.0).
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed of the variance process.
    theta : float
        Long-run mean of the variance process.
    rho : float
        Correlation between asset and variance Brownian motions.
    """

    def __init__(self, u, r, sigma, spot, T, v0, kappa, theta, rho, t=0.0):
        super().__init__(u=u, r=r, sigma=sigma, spot=spot, T=T, t=t)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.rho = rho

    def calculate_phi(self):
        """
        Computes the Heston model characteristic function φ(u).

        Returns
        -------
        complex or np.ndarray
            Value of the characteristic function evaluated at `u`.
        """


def psi(alpha, u, r, sigma, spot, T, t=0, char_func=None, **kwargs):
    """
    Computes the Carr–Madan integrand Ψ(u) for Fourier option pricing.

     This function evaluates the integrand Ψ(u), which is the Fourier transform of the
     exponentially damped European call price. It is used in the Carr–Madan framework to
     recover option prices via inverse Fourier transform (typically implemented with FFT).

     The formula is given by:
        Ψ(u) = [φ(u - i(α + 1))] / [α^2 + α - u^2 + i(2α + 1)u]

     where:
        - φ(u) is the characteristic function of log(S_T)
        - α > 0 is the damping parameter ensuring square-integrability of the payoff
        - u is the complex frequency variable in the Fourier domain

    Input Parameters
    ----------------
    char_func: The characteristic function of the underlying process

    u : complex or np.ndarray
        Frequency domain variable (possibly complex).
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    spot : float
        Spot price of the underlying.
    T : float
        Time to maturity.
    t : float, optional
        Current time (default is 0.0).
    """
    # Basic sanity check for input logic
    if not char_func and not kwargs:
        char_func = BSMCharacteristicFunction

    if char_func and not kwargs:
        raise ValueError("Please input process specific variables")

    # Calculating the modified frequency variable
    u_modified = u - 1j * (alpha + 1)

    # Calculating the characteristic function's value
    char_term = char_func(u_modified, r, sigma, spot, T, t).calculate_phi()

    # Initialising the numerator and denominator terms

    numerator = char_term
    denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u

    return numerator / denominator

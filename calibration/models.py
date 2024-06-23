"""
Implementation of the Heston model for option pricing and calibration.

The Heston model is a stochastic volatility model that describes the evolution
of the underlying asset's price and its variance over time. It is widely used 
in financial mathematics for the pricing of derivatives and risk management.

For detailed theoretical background, refer to the original paper by Steven L. Heston:
`Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility
with Applications to Bond and Currency Options. The Review of Financial Studies, 6(2), 327-343.` 

Link to the paper: https://www.jstor.org/stable/2962057
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")


class HestonCall:
    """
    Implementation of the Heston model for pricing European call options.

    Parameters
    ----------
    S0 : float
        Initial price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option, in years.
    t : int
        Initial time, in years.
    r : float
        Risk-free interest rate.
    v0 : float
        Initial variance of the underlying asset.
    kappa : float
        Rate at which the variance reverts to its long-term mean (theta).
    theta : float
        Long-term mean of the variance.
    sigma : float
        Volatility of the variance (volatility of volatility).
    rho : float
        Correlation between the two Wiener processes that drive the asset price and its variance.
    """

    def __init__(self, S0, K, T, t, r, v0, kappa, theta, sigma, rho):
        """
        Initialisation method for `HestonCall`.
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.t = t
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def characteristic_function(self, phi):
        """
        The characteristic equation of the Fourier transformation.
        """

        # Defining terms for the characteristic equation.

        tau = self.T - self.t  # Calculating the maturity time.
        a = self.kappa * self.theta
        func_C = self.r * phi * 1j * tau + (a / (self.sigma**2))

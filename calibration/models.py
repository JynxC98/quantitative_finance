"""
Implementation of the Heston model for option pricing and calibration.

The Heston model is a stochastic volatility model that describes the evolution
of the underlying asset's price and its variance over time. It is widely used 
in financial mathematics for the pricing of derivatives and risk management.

For detailed theoretical background, refer to the original paper by Steven L. Heston:
Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility
with Applications to Bond and Currency Options. The Review of Financial Studies, 6(2), 327-343.

Link to the paper: https://www.jstor.org/stable/2962057
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")


class HestonCall:
    """
    Implementation of the Heston model for pricing European call options.

    This class provides methods to calculate option prices using the Heston
    stochastic volatility model. It uses the characteristic function approach
    for efficient computation.

    Parameters
    ----------
    S0 : float
        Initial price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option, in years.
    t : float
        Initial time, in years.
    r : float
        Risk-free interest rate (annualized).
    v0 : float
        Initial variance of the underlying asset.
    kappa : float
        Rate at which the variance reverts to its long-term mean (theta).
    theta : float
        Long-term mean of the variance.
    sigma : float
        Volatility of the variance (volatility of volatility).
    rho : float
        Correlation between the two Wiener processes driving the asset price and its variance.
    lambda_ : float
        Variance risk premium.

    Attributes
    ----------
    Same as parameters.

    Notes
    -----
    All parameters should be provided in annual terms where applicable.
    """

    def __init__(self, S0, K, T, t, r, v0, kappa, theta, sigma, rho, lambda_):
        """
        Initialize the HestonCall object with model parameters.

        All parameters are as described in the class docstring.
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
        self.lambda_ = lambda_

    def characteristic_function(self, phi):
        """
        Compute the characteristic function of the log-price process.

        This method implements the closed-form solution for the characteristic
        function of the log-price under the Heston model.

        Parameters
        ----------
        phi : complex
            The argument of the characteristic function.

        Returns
        -------
        complex
            The value of the characteristic function at phi.

        Notes
        -----
        The characteristic function is a key component in the Fourier-based
        pricing method used in the Heston model.
        """
        tau = self.T - self.t  # Time to maturity
        a = self.kappa * self.theta
        b = self.kappa + self.lambda_

        # Complex-valued terms in the characteristic function
        d = np.sqrt(
            ((self.rho * self.sigma * phi * 1j - b) ** 2)
            - (self.sigma**2) * (phi * 1j + phi**2)
        )
        g = (b - self.rho * self.sigma * phi * 1j - d) / (
            b - self.rho * self.sigma * phi * 1j + d
        )

        # Components of the characteristic function
        C = (self.r * phi * 1j * tau) + (a / (self.sigma**2)) * (
            (b - self.rho * self.sigma * phi * 1j - d) * tau
            - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
        )

        D = (
            (b - self.rho * self.sigma * phi * 1j - d)
            / (self.sigma**2)
            * ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
        )

        # The characteristic function
        return np.exp(C + D * self.v0 + 1j * phi * np.log(self.S0))

    # Add other methods with similar detailed docstrings...

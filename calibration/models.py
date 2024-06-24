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
from scipy.integrate import quad

warnings.filterwarnings("ignore")


def heston_char_func(
    x: float,
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    lambda_: float,
    r: float,
    tau: float,
) -> complex:
    """
    Calculate the characteristic function for the Heston model.

    Args:
        x (float): The value at which to evaluate the characteristic function.
        S0 (float): Initial stock price.
        v0 (float): Initial variance.
        kappa (float): Mean reversion speed of variance.
        theta (float): Long-term mean of variance.
        sigma (float): Volatility of variance.
        rho (float): Correlation between stock price and variance.
        lambda_ (float): Market price of volatility risk.
        r (float): Risk-free interest rate.
        tau (float): Time to maturity in years.

    Returns:
        complex: The value of the characteristic function.
    """
    a = kappa * theta
    b = kappa + lambda_
    d = np.sqrt((sigma * rho * x * 1j - b) ** 2 + (sigma**2) * (x * 1j + x**2))
    g = (b - rho * 1j * x * sigma - d) / (b - rho * 1j * x * sigma + d)

    part_1 = (
        np.exp(r * x * 1j * tau)
        * (S0 ** (1j * x))
        * (((1 - g * np.exp(-d * tau)) / (1 - g)) ** (-2 * a / (sigma**2)))
    )

    part_2 = np.exp(
        (a * tau / (sigma**2)) * (b - rho * sigma * 1j * x - d)
        + (v0 / (sigma**2))
        * (b - rho * sigma * 1j * x - d)
        * ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
    )

    return part_1 * part_2


def heston_integrand(
    x: float,
    S0: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    lambda_: float,
    r: float,
    tau: float,
) -> float:
    """
    Calculate the integrand for the Heston model price integral.

    Args:
        x (float): The value at which to evaluate the integrand.
        S0 (float): Initial stock price.
        K (float): Strike price.
        v0 (float): Initial variance.
        kappa (float): Mean reversion speed of variance.
        theta (float): Long-term mean of variance.
        sigma (float): Volatility of variance.
        rho (float): Correlation between stock price and variance.
        lambda_ (float): Market price of volatility risk.
        r (float): Risk-free interest rate.
        tau (float): Time to maturity in years.

    Returns:
        float: The real part of the integrand value.
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambda_, r, tau)
    numerator = np.exp(-1j * x * np.log(K)) * (
        heston_char_func(x - 1j, *args) - K * heston_char_func(x, *args)
    )
    denominator = 1j * x
    return (numerator / denominator).real


def heston_call_price(
    S0: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    lambda_: float,
    r: float,
    tau: float,
) -> float:
    """
    Calculate the call option price using the Heston model.

    Args:
        S0 (float): Initial stock price.
        K (float): Strike price.
        v0 (float): Initial variance.
        kappa (float): Mean reversion speed of variance.
        theta (float): Long-term mean of variance.
        sigma (float): Volatility of variance.
        rho (float): Correlation between stock price and variance.
        lambda_ (float): Market price of volatility risk.
        r (float): Risk-free interest rate.
        tau (float): Time to maturity in years.

    Returns:
        float: The calculated call option price.
    """
    args = (S0, K, v0, kappa, theta, sigma, rho, lambda_, r, tau)
    heston_integral, _ = quad(lambda x: heston_integrand(x, *args), 0, np.inf)
    call_price = 0.5 * (S0 - K * np.exp(-r * tau)) + (1 / np.pi) * heston_integral
    return call_price


if __name__ == "__main__":
    S0 = 100
    v0 = 0.35
    K = 110
    kappa = 1.15
    theta = 0.348
    sigma = 0.3
    rho = -0.64
    lambda_ = 0.03
    r = 0.034
    tau = 1

    args = (S0, K, v0, kappa, theta, sigma, rho, lambda_, r, tau)
    print(heston_call_price(*args))

"""
Pricing of Geometric Asian Options under Heston's Stochastic Volatility Model

This module implements the pricing model described in the paper:
'Pricing of geometric Asian options under Heston's stochastic volatility model'
by Bara Kim and In-Suk Wee.

Reference:
Kim, B., & Wee, I. S. (2011). Pricing of geometric Asian options under Heston's
stochastic volatility model. Quantitative Finance, 11(12), 1795-1811.
https://www.tandfonline.com/doi/abs/10.1080/14697688.2011.596844
"""

import warnings
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")


def geometric_asian_call(
    S0,
    v0,
    theta,
    sigma,
    kappa,
    rho,
    r,
    n,
    T,
    K,
):
    """
    Calculate the price of a geometric Asian call option under Heston's model.

    Args:
        S0 (float): Initial stock price
        v0 (float): Initial volatility
        theta (float): Long-term mean of volatility
        sigma (float): Volatility of volatility
        kappa (float): Mean reversion rate of volatility
        rho (float): Correlation between stock price and volatility
        r (float): Risk-free interest rate
        n (int): Number of terms in series expansion
        T (float): Time to maturity
        K (float): Strike price

    Returns:
        float: Price of the geometric Asian call option
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    call_option = np.exp(-r * T) * (
        (psi(1, 0, *args) - K) / 2 + (1 / np.pi) * geometric_integral(*args, K)
    )
    return np.real(call_option)


def geometric_integral(
    S0,
    v0,
    theta,
    sigma,
    kappa,
    rho,
    r,
    n,
    T,
    K,
) -> float:
    """
    Calculate the integral component of the geometric Asian option price.

    Args:
        Same as geometric_asian_call function

    Returns:
        float: Value of the integral
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T, K)
    option_price, _ = quad(lambda x: integrand(x, *args), 0, np.inf)
    return option_price


def integrand(
    x,
    S0,
    v0,
    theta,
    sigma,
    kappa,
    rho,
    r,
    n,
    T,
    K,
) -> float:
    """
    Calculate the integrand for the geometric Asian option price integral.

    Args:
        x (float): Integration variable
        Other args: Same as geometric_asian_call function

    Returns:
        float: Value of the integrand
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    A = psi(1 + 1j * x, 0, *args)
    B = psi(1j * x, 0, *args)
    C = np.exp(-1j * x * np.log(K)) / (1j * x)
    value = (A - K * B) * C
    return np.real(value)


def psi(
    s,
    w,
    S0,
    v0,
    theta,
    sigma,
    kappa,
    rho,
    r,
    n,
    T,
) -> complex:
    """
    Calculate the characteristic function psi as defined in the paper.

    Args:
        s (complex): Complex argument for the characteristic function
        w (float): Second argument for the characteristic function
        Other args: Same as geometric_asian_call function

    Returns:
        complex: Value of the characteristic function psi
    """
    a1 = 2 * v0 / (sigma**2)
    a2 = 2 * kappa * theta / (sigma**2)
    a3 = (
        np.log(S0)
        + ((r * sigma - kappa * theta * rho) * (T**2)) / (2 * sigma * T)
        - (rho * v0) / sigma
    )
    a4 = np.log(S0) - (rho / sigma) * v0 + (r - ((rho * kappa * theta) / sigma)) * T
    a5 = (kappa * v0 + (kappa**2) * theta * T) / (sigma**2)

    h_matrix = np.zeros(n + 3, dtype=complex)
    h_matrix[2] = 1
    h_matrix[3] = T * (kappa - w * rho * sigma) / 2

    nmat = np.arange(1, n + 1)
    A1 = -(s**2) * (sigma**2) * (1 - rho**2) * T**2
    A2 = s * sigma * T * (sigma - 2 * rho * kappa) - 2 * s * w * sigma**2 * T * (
        1 - rho**2
    )
    A3 = T * (
        kappa**2 * T
        - 2 * s * rho * sigma
        - w * (2 * rho * kappa - sigma) * sigma * T
        - w**2 * (1 - rho**2) * sigma**2 * T
    )

    for i in range(4, n + 3):
        h_matrix[i] = (1 / (4 * (i - 2) * (i - 3))) * (
            A1 * h_matrix[i - 4] + A2 * h_matrix[i - 3] + A3 * h_matrix[i - 2]
        )

    H = np.sum(h_matrix[2:])
    H_tilde = np.dot((nmat / T), h_matrix[3:])

    return np.exp(-a1 * (H_tilde / H) - a2 * np.log(H) + a3 * s + a4 * w + a5)


if __name__ == "__main__":
    # Example usage
    price = geometric_asian_call(
        S0=100,  # Initial stock price
        v0=0.09,  # Initial volatility
        sigma=0.39,  # Volatility of volatility
        theta=0.348,  # Long-term mean of volatility
        kappa=1.15,  # Mean reversion rate
        rho=-0.64,  # Correlation
        r=0.05,  # Risk-free rate
        n=10,  # Number of terms in series expansion
        T=0.2,  # Time to maturity
        K=90,  # Strike price
    )
    print(f"Geometric Asian Call Option Price: {price:.4f}")

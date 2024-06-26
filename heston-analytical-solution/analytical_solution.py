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

import numpy as np
from scipy.integrate import quad


def geometric_asian_call(
    S0, v0, theta, sigma, kappa, rho, r, n, T, K, int_upper_limit=np.inf
):
    """
    Calculate the price of a geometric Asian call option under Heston's stochastic volatility model.

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
        int_upper_limit (float, optional): Upper limit for integration. Defaults to infinity.

    Returns:
        float: Price of the geometric Asian call option
    """
    call = np.exp(-r * T) * (
        (psi(1, 0, S0, v0, theta, sigma, kappa, rho, r, n, T) - K) * 0.5
        + (1 / np.pi)
        * geometric_integral(
            S0, v0, theta, sigma, kappa, rho, r, n, T, K, int_upper_limit
        )
    )
    return np.real(call)


def geometric_integral(S0, v0, theta, sigma, kappa, rho, r, n, T, K, int_upper_limit):
    """
    Calculate the integral component of the geometric Asian option price.

    Args:
        S0, v0, theta, sigma, kappa, rho, r, n, T, K: Same as in geometric_asian_call
        int_upper_limit (float): Upper limit for integration

    Returns:
        float: Value of the integral
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T, K)
    result, _ = quad(integrand, 0, int_upper_limit, args=args)
    return result


def integrand(epsilon, S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Calculate the integrand for the geometric Asian option price integral.

    Args:
        epsilon (float): Integration variable
        S0, v0, theta, sigma, kappa, rho, r, n, T, K: Same as in geometric_asian_call

    Returns:
        float: Value of the integrand
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    A = psi(1 + 1j * epsilon, 0, *args)
    B = psi(1j * epsilon, 0, *args)
    C = np.exp(-1j * epsilon * np.log(K)) / (1j * epsilon)
    return np.real((A - K * B) * C)


def psi(s, w, S0, v0, theta, sigma, kappa, rho, r, n, T):
    """
    Calculate the characteristic function psi as defined in the paper.

    Args:
        s (complex): Complex argument for the characteristic function
        w (float): Second argument for the characteristic function
        S0, v0, theta, sigma, kappa, rho, r, n, T: Same as in geometric_asian_call

    Returns:
        complex: Value of the characteristic function psi
    """
    a1 = 2 * v0 / sigma**2
    a2 = 2 * kappa * theta / sigma**2
    a3 = (
        np.log(S0)
        + ((r * sigma - kappa * theta * rho) * T) / (2 * sigma)
        - (rho * v0) / sigma
    )
    a4 = np.log(S0) - (rho * v0 / sigma) + (r - rho * kappa * theta / sigma) * T
    a5 = (kappa * v0 + kappa**2 * theta * T) / (sigma**2)

    h_matrix = np.zeros(n + 3, dtype=complex)
    h_matrix[2] = 1
    h_matrix[3] = T * (kappa - w * rho * sigma) / 2

    nmat = np.arange(1, n + 1)
    A = 1 / (4 * nmat[1:] * (nmat[1:] - 1))
    B = -(s**2) * (sigma**2) * (1 - rho**2) * T**2
    C = T * (
        s * sigma * T * (sigma - 2 * rho * kappa)
        - 2 * s * w * sigma**2 * T * (1 - rho**2)
    )
    D = T * (
        kappa**2 * T
        - 2 * s * rho * sigma
        - w * (2 * rho * kappa - sigma) * sigma * T
        - w**2 * (1 - rho**2) * sigma**2 * T
    )

    for j in range(4, n + 3):
        h_matrix[j] = A[j - 4] * (
            B * h_matrix[j - 4] + C * h_matrix[j - 3] + D * h_matrix[j - 2]
        )

    H = np.sum(h_matrix[2:])
    H_tilde = np.sum((nmat / T) * h_matrix[3:])

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
        K=100,  # Strike price
    )
    print(f"Geometric Asian Call Option Price: {price:.4f}")

"""
This script is used to generate Legendre nodes and weights for numerical
quadrature.

Author: Harsh Parikh
"""

import numpy as np
from numba import njit


@njit(cache=True)
def LegendrePolynomial(n, x):
    """
    Evaluate Legendre polynomial P_n(x) using recurrence.

    (n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)
    Start: P_0(x) = 1, P_1(x) = x
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x

    p_prev = 1.0
    p_curr = x

    for k in range(1, n):
        p_next = ((2.0 * k + 1.0) * x * p_curr - k * p_prev) / (k + 1.0)
        p_prev = p_curr
        p_curr = p_next

    return p_curr


@njit(cache=True)
def LegendreDerivative(n, x):
    """
    Evaluate the derivative of Legendre polynomial P'_n(x)

    Formula: P'_n(x) = n * (x*P_n(x) - P_{n-1}(x)) / (x^2 - 1)

    Recurrence: (1-x^2)P'_n(x) = n(P_{n-1}(x) - x P_n(x))
    """
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    p_n = LegendrePolynomial(n, x)
    p_n_minus_1 = LegendrePolynomial(n - 1, x)

    return n * (x * p_n - p_n_minus_1) / (x * x - 1.0)


@njit(cache=True)
def findLegendreRoot(n, i, tol=1e-15):
    """
    Find root of Legendre polynomial using Newton-Raphson method.

    Initial guess: cos(pi * (4i - 1) / (4n + 2)) for i-th root
    """
    # Initial approximation using cosine formula
    x = np.cos(np.pi * (4.0 * i - 1.0) / (4.0 * n + 2.0))

    # Newton-Raphson iteration
    for _ in range(100):
        p = LegendrePolynomial(n, x)
        p_prime = LegendreDerivative(n, x)

        dx = p / p_prime
        x -= dx

        if abs(dx) < tol:
            break

    return x


@njit(cache=True)
def generateGaussLegendre(n):
    """
    Generate Gauss-Legendre nodes and weights for n-point quadrature.

    Returns
    -------
    nodes : np.ndarray of shape (n,)
    weights : np.ndarray of shape (n,)
    """
    nodes = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)

    # Roots are symmetric: x_i = -x_{n+1-i}
    for idx in range(n):
        i = idx + 1  # match 1-indexed convention from the C++ version
        x = findLegendreRoot(n, i)

        # Calculate weight using formula: w_i = 2 / ((1-x_i^2) * [P'_n(x_i)]^2)
        p_prime = LegendreDerivative(n, x)
        weight = 2.0 / ((1.0 - x * x) * p_prime * p_prime)

        nodes[idx] = x
        weights[idx] = weight

    return nodes, weights


if __name__ == "__main__":
    nodes, weights = generateGaussLegendre(5)
    print("Nodes:  ", nodes)
    print("Weights:", weights)

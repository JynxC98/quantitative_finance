"""
This script computes the value of an arithmetic Asian option using 
a finite difference method based on the solution of a partial differential equation (PDE). 
An implicit time-stepping scheme is employed to numerically evaluate the option price 
for both call and put variants.

The implementation is based on the following references:

1. https://cs.uwaterloo.ca/research/tr/1996/28/CS-96-28.pdf
2. https://personal.ntu.edu.sg/nprivault/MA5182/asian-options.pdf
3. https://www.iccs-meeting.org/archive/iccs2019/papers/115380317.pdf

Author: Harsh Parikh  
Date: 25th May 2025
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def thomas_algorithm(a, b, c, d, epsilon=1e-6):
    """
    The code has been referenced from this post:
    https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-tdma-aka-thomas-algorithm-using-python-with-nump

    Solve a tridiagonal system using the Thomas algorithm.

    Parameters:
    a: Lower diagonal of the tridiagonal matrix (n-1 elements)
    b: Main diagonal of the tridiagonal matrix (n elements)
    c: Upper diagonal of the tridiagonal matrix (n-1 elements)
    d: Right-hand side of the equation (n elements)
    epslilon: An infinitesimaly small term to avoid zero-division error

    Returns:
    x: Solution vector (n elements)

    Note: The tridiagonal system is of the form:
    [b0 c0  0  0  0]   [x0]   [d0]
    [a1 b1 c1  0  0]   [x1]   [d1]
    [0  a2 b2 c2  0] * [x2] = [d2]
    [0   0 a3 b3 c3]   [x3]   [d3]
    [0   0  0 a4 b4]   [x4]   [d4]
    """

    n = len(d)  # Size of the system
    c_prime = np.zeros(n - 1)  # Modified upper diagonal
    d_prime = np.zeros(n)  # Modified right-hand side

    # Forward sweep: Eliminate lower diagonal
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denominator = b[i] - a[i] * c_prime[i - 1] + epsilon
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denominator

    d_prime[n - 1] = (d[n - 1] - a[n - 1] * d_prime[n - 2]) / (
        b[n - 1] - a[n - 1] * c_prime[n - 2]
    )

    # Backward substitution: Solve for x
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


@jit(nopython=True)
def calculate_price(spot, strike, sigma, r, q, T, M=5000, N=5000, option_type="call"):
    """
    Calculates the price of an arithmetic Asian option using a finite difference method
    for solving the associated PDE with an implicit scheme.

    Parameters:
    ----------
    spot : float
        The current spot price of the underlying asset.
    strike : float
        The strike price of the option.
    sigma : float
        Volatility of the underlying asset (annualized).
    r : float
        Risk-free interest rate (annualized).
    q : float
        Continuous dividend yield (annualized).
    T : float
        Time to maturity in years.
    M : int, optional
        Number of time steps for the PDE grid (default is 5000).
    N : int, optional
        Number of spatial grid steps (for average price dimension) (default is 5000).
    option_type : str, optional
        Type of option to price: "call" or "put" (default is "call").

    Returns:
    -------
    float
        The numerical price of the arithmetic Asian option.
    """

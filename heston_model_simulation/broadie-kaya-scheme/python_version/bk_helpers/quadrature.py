"""
Implementation of Gaussian quadrature using Numpy for Numba's support

Author: Harsh Parikh
"""

import numpy as np
from numba import njit

from .legendre_nodes import generateGaussLegendre


# @njit(cache=True)
def Legendre_Integrate(function, lower_limit, upper_limit, num_nodes=32):
    """
    This function is used to calculate the area under the curve using
    Gauss-Legendre quadrature.

    Input Parameters
    ----------------
    function: The main function to be integrated (must itself be a
        numba-jitted, njit-compatible callable taking and returning a scalar)
    lower_limit: The lower limit of the integral
    upper_limit: The upper limit of the integral
    num_nodes: The number of quadrature nodes for the Legendre polynomial
    """
    nodes, weights = generateGaussLegendre(num_nodes)  # Returns nodes and weights

    result = 0.0

    # Scaling factors for the finite interval [lower_limit, upper_limit]
    half_length = 0.5 * (upper_limit - lower_limit)
    center = 0.5 * (upper_limit + lower_limit)

    # Performing the Gauss-Legendre integration
    for i in range(num_nodes):
        # Mapping the nodes from [-1, 1] to [lower_limit, upper_limit]
        x_mapped = center + half_length * nodes[i]

        # Adding the elements
        result += weights[i] * function(x_mapped)

    # Scaling the result by the half-length of the interval
    return half_length * result


if __name__ == "__main__":

    @njit(cache=True)
    def f(x):
        return x**2

    integral = Legendre_Integrate(f, 0.0, 1.0, 32)
    print("Integral of x^2 on [0,1]:", integral)  # should be ~1/3

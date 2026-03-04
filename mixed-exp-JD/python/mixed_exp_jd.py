"""
Implementation of European option pricing under the Mixed-Exponential 
Jump-Diffusion (MEJD) model.

This script implements the framework proposed in:

    Kou, S. G. (2011). "Option Pricing Under a Mixed-Exponential Jump Diffusion
    Model"

    Management Science.
    http://www.columbia.edu/~sk75/mixedExpManagementSci.pdf

The Mixed-Exponential Jump-Diffusion model extends the classical 
Black-Scholes framework by incorporating jumps with a mixed-exponential 
distribution. This structure preserves analytical tractability while 
allowing flexible approximation of arbitrary jump size distributions.

Author: Harsh Parikh
"""

import numpy as np
from fourier_engine import carr_madan_fourier_engine


def calculate_price_med(spot, strike, sigma, r, T, option_type, **params):
    """
    Parameters
     ----------
     spot : float
         Current spot price of the underlying asset. Must be positive.
     strike : float
         Strike price of the option. Must be positive.
     sigma : float
         Volatility of the diffusion component (annualized). Must be > 0.
     r : float
         Risk-free interest rate (annualized, continuous compounding). Can be negative.
     T : float
         Time to maturity in years (e.g., 0.5 for 6 months). Must be > 0.
     option_type : str
         Type of option to price. Must be either 'call' or 'put'.

     **params : dict, optional
         Mixed-exponential jump parameters. Default values provide a 2-component
         mixture that approximates a double exponential distribution.

         Parameters
         ----------
         lambda_ : float, default=3.0
             Jump intensity (average number of jumps per year). Must be > 0.
             For example, lambda_=3 means on average 3 jumps occur per year.

         p_u : float, default=0.4
             Probability that a jump is upward (positive). Must be in [0,1].
             p_d = 1 - p_u is automatically enforced.

         weights_up : array_like, default=[1.2, -0.2]
             Mixture weights for the upward jump size distribution.
             Each weight can be positive or negative, but must sum to 1.
             Negative weights allow for more flexible distribution fitting.
             Length determines number of mixture components.

         weights_down : array_like, default=[1.3, -0.3]
             Mixture weights for the downward jump size distribution.
             Each weight can be positive or negative, but must sum to 1.
             Negative weights allow for more flexible distribution fitting.
             Length determines number of mixture components.

         scaling_up : array_like, default=[20, 50]
             Scaling factors (η_i⁺) for the upward jump components.
             Each factor must be > 1 to ensure finite expectation.
             Controls the mean size of positive jumps: mean = 1/(η_i⁺ - 1)

         scaling_down : array_like, default=[20, 50]
             Scaling factors (η_j⁻) for the downward jump components.
             Each factor must be > 0.
             Controls the mean size of negative jumps: mean = 1/(η_j⁻ + 1)

     Returns
     -------
     float
         Price of the European option under the MEJD model.

     References
     ----------
     Cai, N., & Kou, S. G. (2011). Option pricing under a mixed-exponential
     jump diffusion model. Management Science, 57(11), 2067-2081.
    """
    # Basic sanity check, ensuring the inputs from the user are accurate
    assert option_type in (
        "call",
        "put",
    ), "Please select an option from `call`or `put`."

    # Fetching the MEM parameters

    lambda_ = params.get("lambda_", 3)

    assert lambda_ >= 0 and np.isreal(
        lambda_
    ), "The value of lambda must be positive and real"

    p_u = params.get("p_u", 0.4)  # Setting the default value to 0.5
    p_d = params.get("p_d", 0.6)  # Setting the default value to 0.5

    # Ensuring the probability conditioning.
    assert np.isclose(p_u + p_d, 1), "The sum of probabilities must be equal to 1"

    # Fetching the weights for the right tail of the returns
    weights_up = np.array(params.get("weights_up", [1.2, -0.2]), dtype=np.float64)

    # Fetching the weights for the left tail of the returns
    weights_down = np.array(params.get("weights_down", [1.3, -0.3]), dtype=np.float64)

    # According to the literature, the sum of all weights must be equal to 1
    assert np.isclose(
        np.sum(weights_up), 1.0
    ), "The sum of all weights must be equal to 1 for the right tail"

    assert np.isclose(
        np.sum(weights_down), 1.0
    ), "The sum of all weights must be equal to 1 for the right tail"

    # Fetching the scaling factors
    scaling_up = np.array(params.get("scaling_up", [20, 50]), dtype=np.float64)

    scaling_down = np.array(params.get("scaling_down", [20, 50]), dtype=np.float64)

    # According to the literature, the scaling factors must:
    # 1. Be greater than 1 for the right tail and greater than 0 for the left tail.
    # 2. The length of the scaling factors must match the length of the respective weights

    # Ensuring the scaling factors are greater than 1 for the right tail
    assert np.all(
        scaling_up > 1
    ), "The scaling factors for the right tail must be greater than 1"
    assert np.all(
        scaling_down > 0
    ), "The scaling factors for the left tail must be greater than 0"

    # Ensuring the shape of weights match the scaling factors.
    assert (
        scaling_up.shape == weights_up.shape
    ), "The lengths of the scaling factors and weights must match for the right tail"

    assert (
        scaling_down.shape == weights_down.shape
    ), "The lengths of the scaling factors and weights must match for the right tail"

    # Calculating the value of `k`as per the literature

    k = (
        p_u * np.sum((weights_up * scaling_up) / (scaling_up - 1))
        + p_d * np.sum((weights_down * scaling_down) / (scaling_down + 1))
    ) - 1

    # Calculating the overall drift of the process.
    mu = r - 0.5 * sigma**2 - lambda_ * k

    # Defining a vectorised implementation of the characteristic function
    def mgf(x):
        """
        This function stores the moment generating function of the MEM process.
        """
        x = np.atleast_1d(np.asarray(x))

        # x shape: (N,), scaling shape: (m,) or (n,)
        # Reshape for broadcasting: (N, 1) vs (1, m)
        x_col = x[:, np.newaxis]  # shape (N, 1)

        # Right tail contribution — shape (N,)
        right_tail = p_u * np.sum(
            (weights_up * scaling_up) / (scaling_up - x_col), axis=1
        )

        # Left tail contribution — shape (N,)
        left_tail = p_d * np.sum(
            (weights_down * scaling_down) / (scaling_down + x_col), axis=1
        )

        result = 0.5 * sigma**2 * x**2 + mu * x + lambda_ * (right_tail + left_tail - 1)

        return result.squeeze()

    def charfunc(x):
        """
        This function stores the characteristic function of the MEM process.
        """
        return np.exp(1j * x * np.log(spot)) * np.exp(T * mgf(1j * x))

    price = carr_madan_fourier_engine(
        spot=spot,
        strike=strike,
        sigma=sigma,
        r=r,
        T=T,
        N=2 << 14,
        char_func=charfunc,
        alpha=1.5,
    )

    if option_type == "call":
        return price
    else:
        return price - (spot - strike * np.exp(-r * T))


if __name__ == "__main__":
    spot = 100
    strike = 100
    sigma = 0.3
    r = 0.05
    T = 1.0
    option_type = "call"

    ## MEM parameters
    params = {
        "lambda_": 0,
        "p_up": 0.4,
        "p_down": 0.6,
        "weights_up": [1.2, -0.2],
        "weights_down": [1.3, -0.3],
        "scaling_up": [20, 50],
        "scaling_down": [20, 50],
    }

    print(calculate_price_med(spot, strike, sigma, r, T, option_type, **params))

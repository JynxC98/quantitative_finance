"""
The main script to calculate the option price using the Carr-Madan Framework

Author: Harsh Parikh
Date: 30th July 2025
"""

import numpy as np

from characteristic_functions import bsm_characteristic_function
from analytical_solutions import bsm_option_price
from fourier_methods import psi


def carr_madan_fourier_engine(spot, strike, sigma, r, T, N, alpha, dv=0.3, t=0):
    """
    Computes the price of a European option using the Carr-Madan FFT method.

    This function implements the Carr–Madan (1999) framework to efficiently compute
    option prices via the Fast Fourier Transform (FFT). The method evaluates the
    damped characteristic function of the log-asset price and recovers option values
    by applying inverse Fourier techniques. This approach is particularly effective
    for pricing across a grid of strikes.

    Input Parameters
    ----------------
    spot: Initial stock price
    strike: Target strike price
    sigma: Volatility of the underlying asset
    r: Risk free rate
    T: Time to maturity
    t: Current time
    dv: Frequency domain spacing
    N: Number of grid points (must be a power of 2)
    alpha: Damping factor

    Returns
    -------
    The computed option price for the specified strike.
    """
    # Creating grid points for the frequency domain
    freq_grid_pts = dv * np.arange(0, N).astype(np.complex128)

    # Creating log-strike spacing
    dk = 2 * np.pi / (N * dv)  # Nyquist-Shannon condition to avoid aliasing
    b = 0.5 * N * dk  # The `b` term in log-strike spacing

    # Computing the log-strike domain
    log_strike_grid = -b + np.arange(0, N) * dk

    # Computing the `Psi` grid
    psi_grid = np.zeros((N), dtype=np.complex128)

    # Computing the trapezoidal weights
    weights = np.ones((N))
    weights[0] = 0.5  # Weight corresponding to index `0`
    weights[-1] = 0.5  # Weight corresponding to index `N-1`

    # Calculating the Psi grid
    psi_inputs = (
        r,
        sigma,
        spot,
        T,
    )  # The order of the parameters should match the corresponding characteristic function's input

    psi_grid = (
        np.exp(-r * (T - t))
        * dv
        * psi(alpha, freq_grid_pts, bsm_characteristic_function, psi_inputs)
        * np.exp(1j * b * freq_grid_pts)
        * weights
    )

    # Reverting the complex representation of the call prices vector
    call_prices = np.fft.fft(psi_grid)

    # Fetching the strikes grid
    strikes = np.exp(log_strike_grid)

    # Calculating the call price output
    call_price_output = (np.exp(-alpha * log_strike_grid) / np.pi) * call_prices.real

    price = np.interp(strike, strikes, call_price_output)

    return price


if __name__ == "__main__":

    # Option properties
    spot = 100
    strike = 110
    T = 1.0
    sigma = 0.25
    r = 0.035

    # Carr-Madan engine properties
    alpha = 1
    N = 1 << 15

    # Calculating the Carr-Madan price
    carr_madan_price = carr_madan_fourier_engine(spot, strike, sigma, r, T, N, alpha)

    # Calculating the BS< price
    bsm_price = bsm_option_price(spot, strike, sigma, r, T)

    # Printing the Carr-Madan price
    print(f"The Carr-Madan price is {carr_madan_price}")

    # Printing the BSM price
    print(f"The BSM price is {bsm_price}")

    # Printing the pricing error
    print(f"Pricing error: {np.absolute(bsm_price - carr_madan_price)}")

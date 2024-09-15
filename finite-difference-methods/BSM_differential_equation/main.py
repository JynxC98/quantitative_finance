"""
Main script to run the FDM methods
"""
import numpy as np

from models import (
    explicit_scheme,
    implicit_scheme,
    black_scholes_merton
)
def run_script():
    # Set parameters
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1.0  # Time to maturity
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    Smax = 2 * S  # Maximum stock price
    M = 100  # Number of stock price steps
    N = 1000  # Number of time steps
    dS = Smax / 100
    S_range = np.linspace(dS, Smax, M+1)

    # Calculatint the analytical values of the option prices at t = 0
    analytical_call = black_scholes_merton(S_range, K, T, r, sigma, 'call')
    analytical_put = black_scholes_merton(S_range, K, T, r, sigma, 'put')

    # Approximating prices at t = 0 
    explicit_call, _ = explicit_scheme( Smax, K, T, r, sigma, M, N, 'call')
    explicit_put, _ = explicit_scheme( Smax, K, T, r, sigma, M, N, 'put')
    implicit_call, _ = implicit_scheme( Smax, K, T, r, sigma, M, N, 'call')
    implicit_put, _ = implicit_scheme( Smax, K, T, r, sigma, M, N, 'put')

    explicit_call_error = np.abs(explicit_call[:, 0] - analytical_call).mean()
    explicit_put_error = np.abs(explicit_put[:, 0] - analytical_put).mean()
    implicit_call_error = np.abs(implicit_call[:, 0] - analytical_call).mean()
    implicit_put_error = np.abs(implicit_put[:, 0] - analytical_put).mean()

    print(f"Mean Absolute Error (Explicit Call): {explicit_call_error:.6f}")
    print(f"Mean Absolute Error (Explicit Put): {explicit_put_error:.6f}")
    print(f"Mean Absolute Error (Implicit Call): {implicit_call_error:.6f}")
    print(f"Mean Absolute Error (Implicit Put): {implicit_put_error:.6f}")



if __name__ =="__main__":
    run_script()

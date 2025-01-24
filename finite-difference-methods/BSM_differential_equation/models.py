"""
A Python script to perform finite difference analysis.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_merton(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the price of a European option using the Black-Scholes-Merton formula.

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate (annualized)
    sigma: Volatility of the underlying stock (annualised)
    option_type: 'call' for Call option, 'put' for Put option

    Returns:
    price: Option price
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the option price based on the option type
    if option_type == "call":
        # Call option price
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        # Put option price
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def explicit_scheme(Smax, K, T, r, sigma, M, N, option_type="call"):
    """
    Implement the explicit finite difference scheme for option pricing.

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free interest rate
    sigma: Volatility
    Smax: Maximum stock price in the grid
    M: Number of stock price steps
    N: Number of time steps
    option_type: 'call' or 'put'

    Returns:
    grid: Option price grid
    S_values: Array of stock prices
    """

    # Calculate step sizes
    dt = T / N  # Time step
    dS = Smax / M  # Stock price step

    # Initialise grid
    grid = np.zeros(
        (M + 1, N + 1)
    )  # (M+1) x (N+1) grid for stock prices and time steps
    S_values = np.linspace(dS, Smax, M + 1)  # Array of stock prices

    # Terminal condition (option payoff at expiry)
    if option_type == "call":
        grid[:, -1] = np.maximum(S_values - K, 0)
    else:
        grid[:, -1] = np.maximum(K - S_values, 0)

    # Boundary conditions
    if option_type == "call":
        grid[0, :] = 0  # Option value is 0 when S = 0
        grid[-1, :] = Smax - K * np.exp(
            -r * np.linspace(0, T, N + 1)
        )  # S = Smax boundary
    else:
        grid[0, :] = K * np.exp(-r * np.linspace(0, T, N + 1))  # S = 0 boundary
        grid[-1, :] = 0  # Option value is 0 when S = Smax for put options

    # Iterate backwards in time
    for n in range(N - 1, -1, -1):
        for i in range(1, M):  # Avoid boundaries (i=0 and i=I)
            # Calculate coefficients for the explicit scheme
            # These coefficients come from the discretisation of the Black-Scholes PDE
            # V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0

            # Coefficient for V_{i-1,n+1}
            a = 0.5 * dt * (sigma**2 * i**2 - r * i)

            # Coefficient for V_{i,n+1}
            b = 1 - dt * (sigma**2 * i**2 + r)

            # Coefficient for V_{i+1,n+1}
            c = 0.5 * dt * (sigma**2 * i**2 + r * i)

            # Update grid point using explicit scheme formula
            grid[i, n] = (
                a * grid[i - 1, n + 1] + b * grid[i, n + 1] + c * grid[i + 1, n + 1]
            )

    return grid, S_values


def implicit_scheme(Smax, K, T, r, sigma, M, N, option_type="call"):
    """
    Implement the implicit finite difference scheme for option pricing.

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free interest rate
    sigma: Volatility
    Smax: Maximum stock price in the grid
    I: Number of stock price steps
    N: Number of time steps
    option_type: 'call' or 'put'

    Returns:
    grid: Option price grid
    S_values: Array of stock prices
    """

    # Calculate step sizes
    dt = T / N  # Time step
    dS = Smax / M  # Stock price step

    # Initialize grid
    grid = np.zeros(
        (M + 1, N + 1)
    )  # (I+1) x (N+1) grid for stock prices and time steps
    S_values = np.linspace(0, Smax, M + 1)  # Array of stock prices

    # Set terminal condition (option payoff at expiry)
    if option_type == "call":
        grid[:, N] = np.maximum(S_values - K, 0)
    else:
        grid[:, N] = np.maximum(K - S_values, 0)

    # Set boundary conditions
    if option_type == "call":
        grid[0, :] = 0  # Option value is 0 when S = 0
        grid[M, :] = Smax - K * np.exp(
            -r * np.linspace(0, T, N + 1)
        )  # S = Smax boundary
    else:
        grid[0, :] = K * np.exp(-r * np.linspace(0, T, N + 1))  # S = 0 boundary
        grid[M, :] = 0  # Option value is 0 when S = Smax for put options

    # Precompute coefficients for the tridiagonal system
    a = np.zeros(M - 1)  # Lower diagonal
    b = np.zeros(M - 1)  # Main diagonal
    c = np.zeros(M - 1)  # Upper diagonal

    # Coefficients for V_{n+1, i-1}
    a = 0.5 * dt * (sigma**2 * S_values**2 / dS**2 - r * S_values / dS)

    # Coefficients for V_{n+1, i}
    b = -(1 + dt * (sigma**2 * S_values**2 / dS**2 + r))

    # Coefficients for V_{n+1, i+1}
    c = 0.5 * dt * (sigma**2 * S_values**2 / dS**2 + r * S_values / dS)

    # Iterate backwards in time (from maturity to present)
    for n in range(N - 1, -1, -1):
        # Set up the right-hand side (RHS) of the linear system
        d = -grid[1:-1, n + 1]

        # Adjust the boundary conditions in the RHS
        d[0] -= a[0] * grid[0, n]
        d[-1] -= c[-1] * grid[M, n]

        # Use the Thomas algorithm to solve the tridiagonal system
        grid[1:-1, n] = thomas_algorithm(a, b, c, d)

    return grid, S_values


def thomas_algorithm(a, b, c, d):
    """
    The code has been referenced from this post:
    https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-tdma-aka-thomas-algorithm-using-python-with-nump

    Solve a tridiagonal system using the Thomas algorithm.

    Parameters:
    a: Lower diagonal of the tridiagonal matrix (n-1 elements)
    b: Main diagonal of the tridiagonal matrix (n elements)
    c: Upper diagonal of the tridiagonal matrix (n-1 elements)
    d: Right-hand side of the equation (n elements)

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
        denominator = b[i] - a[i] * c_prime[i - 1]
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

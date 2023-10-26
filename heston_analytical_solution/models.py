import numpy as np
from scipy.integrate import quad


def geometric_asian_call(S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    S0: Initial stock price
    v0: Initial volatility
    theta: long run average of the volatility
    sigma: volatility of the volatility
    kappa: rate of mean reversion
    rho: correlation between the two Brownian motions
    r: risk free rate
    T: Time to maturity
    K: Strike price
    """


def geometric_integral(S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Params as same as `geometric_asian_call` function.
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T, K)
    option_price, _ = quad(
        lambda x: 0.5 + (1 / np.pi) * np.real(integrand(x, *args)), 0, 1e5
    )

    return option_price


def integrand(x, S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Params as same as `geometic_asian_call` function.
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    A = psi(1 + 1j * x, 0, *args)
    B = psi(1j, 0, *args)
    C = np.exp(-1j * x * np.log(K)) / (1j * x)
    value = (A / B) * C
    return value


def psi(s, w, S0, v0, theta, sigma, kappa, rho, r, n, T):
    """
    s, w: complex numbers. \\
    Properties of s and w: \\
    1. Real(s) >= 0 \\
    2. Real(w) >= 0 \\
    3. 0 <= Real(s) + Real(w) <= 1 \\
    """
    a1 = 2 * v0 / (sigma**2)
    a2 = 2 * kappa * theta / (sigma**2)
    a3 = (
        np.log(S0)
        + ((r * sigma - kappa * theta * rho) * (T**2) / (2 * sigma * T))
        - (rho * T / (sigma * T)) * v0
    )
    a4 = np.log(S0) - (rho / sigma) * v0 + (r - ((rho * kappa * theta) / sigma)) * T
    a5 = (kappa * v0 + kappa**2 * theta * T) / (sigma**2)
    h_matrix = np.zeros([n + 3, 1])
    h_matrix[2] = 1  # H_0 = 1, saving for the third entry.
    h_matrix[3] = T * (kappa - w * rho * sigma) / 2  # H1, will save in the 4th entry.

    # Calculating from H2 to H_n
    nmat = np.arange(1, n + 1).T

    constants = np.array([1 / (4 * num * (num - 1)) for num in np.arange(2, n + 5)])
    A1 = -(s**2) * (sigma**2) * (1 - rho**2) * T**2
    A2 = (
        s * sigma * T * (sigma - 2 * rho * kappa)
        - 2 * s * w * sigma**2 * T * (1 - rho**2) * T
    )
    A3 = T * (
        kappa**2 * T
        - 2 * s * rho * sigma
        - w * (2 * rho * kappa - sigma) * sigma * T
        - w**2 * (1 - rho**2) * sigma**2 * T
    )
    for i in range(4, n + 3):
        h_matrix[i] = constants[i - 4] * (
            A1 * h_matrix[i - 4] + A2 * h_matrix[i - 3] + A3 * h_matrix[i - 2]
        )
    H = np.sum(h_matrix, 1)
    H_tilde = np.dot(nmat, h_matrix[3:])

    return np.exp(-a1 * (H_tilde / H) - a2 * np.log(H) + a3 * s + a4 * w + a5)


if __name__ == "__main__":
    # value = define(1+1i, 0, 100, 0.09, 0.348, 0.39,1.15, -0.64, 0.05, 10, 1);
    value = psi(
        s=1 + 1j,
        w=0,
        S0=100,
        v0=0.09,
        theta=0.348,
        kappa=1.15,
        rho=-0.64,
        r=0.05,
        n=10,
        T=1,
        sigma=0.39,
    )
    test_2 = geometric_integral(
        S0=100,
        v0=0.09,
        sigma=0.39,
        theta=0.348,
        kappa=1.15,
        rho=-0.64,
        r=0.05,
        n=10,
        T=1,
        K=90,
    )
    print(value)

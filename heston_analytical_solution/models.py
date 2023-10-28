import numpy as np
from scipy.integrate import quad


def geometric_asian_call(S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Calculates the price of the call option.
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    call_option = np.exp(-r * T) * (
        (psi(1, 0, *args) - K) / 2 + (1 / np.pi) * geometric_integral(*args, K)
    )
    return np.real(call_option)


def geometric_integral(S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Params as same as `geometric_asian_call` function.
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T, K)
    option_price, _ = quad(lambda x: integrand(x, *args), 0, 10e5)

    return option_price


def integrand(x, S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Params as same as `geometic_asian_call` function.
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T)
    A = psi(1 + 1j * x, 0, *args)
    B = psi(1j * x, 0, *args)
    C = np.exp(-1j * x * np.log(K)) / (1j * x)
    value = (A - K * B) * C
    return np.real(value)


def psi(s, w, S0, v0, theta, sigma, kappa, rho, r, n, T):
    a1 = 2 * v0 / (sigma**2)
    a2 = 2 * kappa * theta / (sigma**2)
    a3 = (
        np.log(S0)
        + ((r * sigma - kappa * theta * rho) * (T**2) / (2 * sigma * T))
        - (rho * T / (sigma * T)) * v0
    )
    a4 = np.log(S0) - (rho / sigma) * v0 + (r - ((rho * kappa * theta) / sigma)) * T
    a5 = (kappa * v0 + kappa**2 * theta * T) / (sigma**2)
    h_matrix = np.zeros(
        [n + 3], dtype=complex
    )  # n + 3 because we need to store values from h_(n-2) to h_n

    h_matrix[2] = 1
    h_matrix[3] = T * (kappa - w * rho * sigma) / 2
    nmat = np.arange(1, n + 1)

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
        h_matrix[i] = (1 / (4 * (i - 2) * (i - 3))) * (
            A1 * h_matrix[i - 4] + A2 * h_matrix[i - 3] + A3 * h_matrix[i - 2]
        )
        # print(h_matrix[i])
    H = np.sum(h_matrix)
    H_tilde = np.dot(nmat, h_matrix[3:])
    return np.exp(-a1 * (H_tilde / H) - a2 * np.log(H) + a3 * s + a4 * w + a5)


if __name__ == "__main__":
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
    test_2 = geometric_asian_call(
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
    print(test_2)

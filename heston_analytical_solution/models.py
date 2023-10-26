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

    pass


def integrand(x, S0, v0, theta, sigma, kappa, rho, r, n, T, K):
    """
    Params as same as `geometic_asian_call` function.
    """
    args = (S0, v0, theta, sigma, kappa, rho, r, n, T, K)
    # A=psi(1+1i*x,0,S0,v0,theta,sigma,kappa,rho,r,n,T);
    # B=psi(x*1i,0,S0,v0,theta,sigma,kappa,rho,r,n,T);
    # C=exp(-1i*x*log(K))./(1i*x);
    # value=real((A-K.*B).*C); %return the real part only
    # end

    w = 0
    A = (1 + 1j,)
    pass


def psi(s, w, S0, v0, theta, sigma, kappa, rho, r, n, T, K):
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
    a5 = (kappa * v0 + kappa**2 * T) / (sigma**2)

    h_matrix = np.zeros([n + 2, 1])
    h_matrix[2, :] = 1  # H_0 = 1, saving for the third entry.
    h_matrix[3, :] = (
        T * (kappa - w * rho * sigma) / 2
    )  # H1, will save in the 4th entry.

    # Calculating from H2 to H_n
    nmat = np.linspace(1, n, n).reshape(-1, 1)
    A1 = 1 / (4 * nmat[1:] * (nmat[1:] - 1))
    A2 = -(s**2) * sigma * (1 - rho**2) * (T**2)
    A3 = (s * sigma * T * (sigma - 2 * rho * kappa)) - 2 * s * w * (sigma**2) * T * (
        1 - (rho**2)
    )
    A4 = T * (
        (kappa**2) - 2 * s * rho * sigma - w * (2 * rho * kappa - sigma)
    ) * sigma * T - ((w**2) * (1 - (rho**2)) * (sigma**2) * T)
    for j in range(4, n + 2):
        h_matrix[j, :] = A1[j - 2, 1] * A2 * h_matrix[j - 2, :] + A3 * (
            T * h_matrix[j - 1, :] + A4 * h_matrix[j - 1, :]
        )
    H = np.sum(h_matrix[2:, :], 1)
    h_tilde = nmat / T * h_matrix[3:, :]
    H_tilde = np.sum(h_tilde, 1)
    return np.exp(-a1 * H_tilde / H - a2 * np.log(H) + a3 * s + a4 * w + a5)


if __name__ == "__main__":
    value = psi(
        s=1 + 1j,
        w=0,
        S0=100,
        v0=0.09,
        theta=0.09,
        kappa=2.0,
        rho=-0.3,
        r=0.05,
        n=10,
        T=1,
        K=100,
        sigma=0.2,
    )
    print(value)

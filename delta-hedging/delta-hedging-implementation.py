import numpy as np
from scipy.stats import norm
from black_scholes import call_option_price, put_option_price


def delta_hedge(S0, r, T, n, N, mu, sigma, K, option_type="call"):
    option_params = (S0, K, T, r, sigma)
    n_steps = 2**n
    if option_type == "call":
        option_price = call_option_price(*option_params)
    else:
        option_price = put_option_price(*option_params)

    time_steps = T / n_steps
    time_discretisation = np.linspace(0, T, n_steps + 1).reshape(-1, 1)
    dW = np.zeros([n_steps + 1, N])
    dW[1 : n_steps + 1, :] = np.random.normal(0, 1, size=(n_steps, N)) * np.sqrt(
        time_steps
    )

    W = np.cumsum(dW, axis=0)
    time_grid = time_discretisation.reshape(-1, 1) * np.ones((n_steps + 1, N))
    increment = sigma * W + (mu - 0.5 * (sigma**2)) * time_grid
    St = S0 * np.exp(-r * increment)
    time_differences = (T - time_discretisation) * np.ones((n_steps + 1, N))
    discount_factor_matrix = np.exp(-r * time_grid)

    d1 = np.log(
        St[1 : n_steps + 1, :] / (K * np.exp(-r * time_differences[1 : n_steps + 1, :]))
        + 0.5 * (r + sigma**2) * time_differences[1 : n_steps + 1, :]
    ) / (sigma * np.sqrt(time_differences[1 : n_steps + 1, :]))
    delta = norm.cdf(d1)
    price_difference = (
        St[1 : n_steps + 1, :] * discount_factor_matrix[1 : n_steps + 1, :]
        - St[:n_steps, :] * discount_factor_matrix[:n_steps, :]
    )
    sum_holding = np.sum(delta * price_difference, axis=0)
    X = np.exp(-r * T) * (option_price - sum_holding)

    upayoff = St[n_steps, :] - K
    payoff = upayoff * (upayoff > 0)
    PNL = X - payoff
    return PNL, X


if __name__ == "__main__":
    profit, holding = delta_hedge(
        S0=100, r=0.05, T=1, n=5, N=1000, mu=0.02, sigma=0.2, K=80
    )
    print(
        f"Overall Profit is {profit.mean()} and the Value of the portfolio is {holding.mean()}"
    )

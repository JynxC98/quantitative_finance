"""
Codes to simulate asset prices (GBM) and Heston model for stochastic volatility.
Uses Euler scheme
"""

import time
import numpy as np


def geometric_asian_call_price_euler(**kwargs):
    """
    Simulation of Heston Model under euler scheme.

    Input Parameters
    ----------------
    S0: Stock price at t = 0
    v0: Volatility at t = 0
    theta: long run average of the volatility
    sigma: volatility of volatility
    kappa: rate of mean reversion
    rho: correlation between two brownian motions
    r: Interest rate of riskless assets
    T: Option's contract period. Measured in years.
    K: Option's strike price
    num_paths: Number of paths in Brownian motion
    step_size: Time Increments

    Returns
    -------
    Euler call option price
    Euler confidence interval
    """
    S0 = kwargs.get("S0", 100)
    v0 = kwargs.get("v0", 0.09)
    theta = kwargs.get("theta", 0.348)
    sigma = kwargs.get("sigma", 0.39)
    kappa = kwargs.get("kappa", 1.15)
    rho = kwargs.get("rho", -0.64)
    r = kwargs.get("r", 0.05)
    T = kwargs.get("T", 0.2)
    K = kwargs.get("K", 90)
    num_paths = kwargs.get("num_paths", 5000)
    step_size = kwargs.get("step_size", 1e-3)
    N = int(T / step_size)

    stock_prices = np.zeros((num_paths, N + 1))
    stock_prices[:, 0] = S0
    sigma_v = v0 * np.ones(num_paths)

    np.random.seed(1)

    for n in range(1, N + 1):
        dW1 = np.random.randn(num_paths) * np.sqrt(step_size)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.randn(num_paths) * np.sqrt(
            step_size
        )

        stock_prices[:, n] = stock_prices[:, n - 1] * (
            1 + r * step_size + np.sqrt(np.abs(sigma_v)) * dW1
        )
        sigma_v = (
            sigma_v
            + kappa * (theta - sigma_v) * step_size
            + sigma * np.sqrt(np.abs(sigma_v)) * dW2
        )

    geometric_avg = np.exp(np.mean(np.log(stock_prices[:, 1:]), axis=1))
    payoff = np.maximum(geometric_avg - K, 0)
    discounted_payoff = np.exp(-r * T) * payoff

    call_euler = np.mean(discounted_payoff)
    std_payoff = np.std(discounted_payoff)

    conf_left = call_euler - 1.96 * std_payoff / np.sqrt(num_paths)
    conf_right = call_euler + 1.96 * std_payoff / np.sqrt(num_paths)
    euler_confidence = (conf_left, conf_right)

    return {
        "Mean Call Option Price": call_euler,
        "Confidence Interval": euler_confidence,
    }


def simulate_heston_model_milstein(**kwargs) -> dict:
    """
    Simulation of Heston Model under Milstein scheme.

    Input Parameters
    ----------------
    S0: Stock price at t = 0
    v0: Volatility at t = 0
    theta: long run average of the volatility
    sigma: volatility of volatility
    kappa: rate of mean reversion
    rho: correlation between two brownian motions
    r: Interest rate of riskless assets
    T: Option's contract period. Measured in years.
    K: Option's strike price
    num_paths: Number of paths in Brownian motion
    step_size: Time Increments

    Returns
    -------
    Milstein call option price
    Milstein confidence interval
    """
    S0 = kwargs.get("S0", 100)
    v0 = kwargs.get("v0", 0.09)
    r = kwargs.get("r", 0.05)
    theta = kwargs.get("theta", 0.348)
    rho = kwargs.get("rho", -0.64)
    kappa = kwargs.get("kappa", 1.15)
    sigma = kwargs.get("sigma", 0.39)
    r = kwargs.get("r", 0.05)
    T = kwargs.get("T", 1)
    K = kwargs.get("K", 90)
    num_paths = kwargs.get("num_paths", 500000)
    step_size = kwargs.get("step_size", 10e-3)

    num_iterations = int(T / step_size)

    stock_price = S0 * np.ones([num_paths, 1])
    volatility = v0 * np.ones([num_paths, 1])

    # Seed generates the same brownian motion
    np.random.seed(1)

    for _ in range(num_iterations):
        dW1 = np.random.randn(num_paths, 1) * np.sqrt(step_size)

        dW2 = rho * dW1 + np.sqrt(1 - pow(rho, 2)) * np.random.randn(
            num_paths, 1
        ) * np.sqrt(step_size)

        # To find the next stock price, we need previous volatility.
        stock_price = stock_price * (
            1 + (r) * step_size + np.sqrt(np.abs(volatility)) * dW1
        ) + (0.5) * volatility * 2 * step_size * (dW1 * 2 - 1)
        volatility = volatility + (
            kappa * (theta - volatility) * step_size
            + sigma * np.sqrt(np.abs(volatility)) * dW2
        )

    payoff = np.exp(-r * T) * (np.maximum(stock_price - K, 0))

    std_payoff = np.std(payoff)
    call_euler = np.mean(payoff)
    v_left = call_euler - 1.96 * std_payoff / np.sqrt(num_paths)
    v_right = call_euler + 1.96 * std_payoff / np.sqrt(num_paths)
    confidence_interval = tuple([v_left, v_right])

    return {
        "Mean Call Option Price": call_euler,
        "Confidence Interval": confidence_interval,
    }


if __name__ == "__main__":
    start_time = time.time()
    print(
        geometric_asian_call_price_euler(
            S0=100,
            v0=0.09,
            theta=0.348,
            sigma=0.39,
            kappa=1.15,
            rho=-0.64,
            r=0.05,
            T=0.2,
            K=90,
            num_paths=10000,
            step_size=1e-3,
        )
    )
    end_time = time.time()
    print(f"Time taken = {end_time - start_time}")

"""
Codes to simulate asset prices (GBM) and Heston model for stochastic volatility.
Uses Euler scheme
"""

import time
import numpy as np


def simulate_heston_model_euler(**kwargs) -> dict:
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
    T = kwargs.get("T", 1)
    K = kwargs.get("K", 90)
    num_paths = kwargs.get("num_paths", 500000)
    step_size = kwargs.get("step_size", 10e-3)

    num_iterations = int(T / step_size)
    total_stock_price = S0 * np.ones([num_paths, 1])

    stock_price = S0 * np.ones([num_paths, 1])
    volatility = v0 * np.ones([num_paths, 1])

    # Seed generates the same brownian motion
    np.random.seed(1)

    for _ in range(num_iterations):
        dZ = np.random.randn(num_paths, 1) * np.sqrt(step_size)

        dW = rho * dZ + np.sqrt(1 - pow(rho, 2)) * np.random.randn(
            num_paths, 1
        ) * np.sqrt(step_size)

        # To find the next stock price, we need previous volatility.
        stock_price = stock_price * (
            1 + (r) * step_size + np.sqrt(np.abs(volatility)) * dW
        )
        volatility = volatility + (
            kappa * (theta - volatility) * step_size
            + sigma * np.sqrt(np.abs(volatility)) * dW
        )

        total_stock_price += stock_price

    mean_stock_price = total_stock_price / num_iterations

    payoff = np.exp(-r * T) * (np.maximum(mean_stock_price - K, 0))

    std_payoff = np.std(payoff)
    call_euler = np.mean(payoff)
    v_left = call_euler - 1.96 * std_payoff / np.sqrt(num_paths)
    v_right = call_euler + 1.96 * std_payoff / np.sqrt(num_paths)
    confidence_interval = tuple([v_left, v_right])

    return {
        "Mean Call Option Price": call_euler,
        "Confidence Interval": confidence_interval,
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
        simulate_heston_model_euler(
            S0=100,
            v0=0.09,
            theta=0.348,
            sigma=0.39,
            kappa=1.15,
            rho=-0.64,
            r=0.05,
            T=1,
            K=90,
            num_paths=50000,
            step_size=10e-4,
        )
    )
    end_time = time.time()
    print(f"Time taken = {end_time - start_time}")

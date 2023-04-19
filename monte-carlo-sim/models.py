"""
Codes to simulate asset prices (GBM) and Heston model for stochastic volatility.
Uses Euler scheme
"""
import numpy as np


def simulate_heston_model_euler(**kwargs) -> dict:
    """
    Simulation of Heston Model under euler scheme.

    Input Parameters
    ----------------
    S_0: Stock price at t = 0
    v_0: Volatility at t = 0
    theta: long run average of the volatility
    sigma: volatility of volatility
    kappa: rate of mean reversion
    rho: correlation between two brownian motions
    risk_free_rate: Interest rate of riskless assets
    time_to_maturity: Option's contract period. Measured in years.
    strike_price: Option's strike price
    num_paths: Number of paths in Brownian motion
    step_size: Time Increments

    Returns
    -------
    Euler call option price
    Euler confidence interval
    """
    S_0 = kwargs.get("S_0", 100)
    v_0 = kwargs.get("v_0", 0.09)
    theta = kwargs.get("theta", 0.348)
    sigma = kwargs.get("sigma", 0.39)
    kappa = kwargs.get("kappa", 1.15)
    rho = kwargs.get("rho", -0.64)
    risk_free_rate = kwargs.get("risk_free_rate", 0.05)
    time_to_maturity = kwargs.get("time_to_maturity", 1)
    strike_price = kwargs.get("strike_price", 90)
    num_paths = kwargs.get("num_paths", 500000)
    step_size = kwargs.get("step_size", 10e-3)

    if len(kwargs) < 11:
        raise ValueError("Have you entered all the arguments? \n")

    num_iterations = int(time_to_maturity / step_size)
    total_stock_price = S_0 * np.ones([num_paths, 1])

    stock_price = S_0 * np.ones([num_paths, 1])
    volatility = v_0 * np.ones([num_paths, 1])

    # Seed generates the same brownian motion
    np.random.seed(1)

    for _ in range(num_iterations):
        dW1 = np.random.randn(num_paths, 1) * np.sqrt(step_size)

        dW2 = rho * dW1 + np.sqrt(1 - pow(rho, 2)) * np.random.randn(
            num_paths, 1
        ) * np.sqrt(step_size)

        # To find the next stock price, we need previous volatility.
        stock_price = stock_price * (
            1 + (risk_free_rate) * step_size + np.sqrt(np.abs(volatility)) * dW1
        )
        volatility = volatility + (
            kappa * (theta - volatility) * step_size
            + sigma * np.sqrt(np.abs(volatility)) * dW2
        )

        total_stock_price += stock_price

    mean_stock_price = total_stock_price / num_iterations

    payoff = np.exp(-risk_free_rate * time_to_maturity) * (
        np.maximum(mean_stock_price - strike_price, 0)
    )

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
    print(
        simulate_heston_model_euler(
            S_0=100,
            v_0=0.09,
            theta=0.348,
            sigma=0.39,
            kappa=1.15,
            rho=-0.64,
            risk_free_rate=0.05,
            time_to_maturity=1,
            strike_price=90,
            num_paths=500000,
            step_size=10e-4,
        )
    )

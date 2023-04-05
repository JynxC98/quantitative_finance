"""
Codes to simulate asset prices (GBM) and Heston model for stochastic volatility.
Uses Euler scheme
"""
import numpy as np


def simulate_heston_model_euler(
    S_0,
    v_0,
    theta,
    sigma,
    kappa,
    rho,
    risk_free_rate,
    time_to_maturity,
    strike_price,
    num_paths,
    step_size,
) -> dict:
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

    num_iterations = int(time_to_maturity / step_size)
    total_stock_price = S_0 * np.ones([num_paths, 1])

    stock_price = S_0 * np.ones([num_paths, 1])
    volatility = v_0 * np.ones([num_paths, 1])

    for _ in range(1, num_iterations + 1):
        dW1 = np.random.normal(0, 1, size=(num_paths, 1)) * np.sqrt(step_size)

        dW2 = rho * dW1 + np.sqrt(1 - pow(rho, 2)) * np.random.normal(
            0, 1, size=(num_paths, 1)
        ) * np.sqrt(step_size)

        # To find the next stock price, we need previous volatility.
        stock_price = stock_price * (
            (1 + risk_free_rate) * step_size + np.sqrt(abs(volatility)) * dW1
        )
        volatility = volatility + (
            kappa * (theta - volatility) * step_size
            + sigma * np.sqrt(abs(volatility)) * dW2
        )

        total_stock_price += stock_price

    mean_stock_price = total_stock_price / num_iterations
    print(mean_stock_price[0:5])

    payoff = np.exp(-risk_free_rate * time_to_maturity) * (
        np.maximum(mean_stock_price - strike_price, 0)
    )

    std_payoff = np.std(payoff)
    call_euler = np.mean(payoff)
    confidence_interval = tuple([(call_euler - std_payoff), (call_euler + std_payoff)])

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
            num_paths=5000,
            step_size=10e-3,
        )
    )

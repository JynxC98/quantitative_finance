"""
Binomial method of option pricing.

Author: Harsh Parikh
"""

import numpy as np


def binomial_method(spot, strike, r, sigma, T, N=100, option_type="call"):
    """
    Prices a European option using the binomial tree method.

    Parameters:
    - spot: Current spot price of the underlying asset
    - strike: Strike price of the option
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility of the underlying asset (annualized)
    - T: Time to maturity in years
    - N: Number of time steps
    - option_type: 'call' or 'put'

    Returns:
    - Option price
    """
    # Calculating the time step
    dt = T / N

    # Calculating the up factor
    u = np.exp(sigma * np.sqrt(dt))

    # Calculating the down factor
    d = 1 / u

    # Calculating the risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialising asset prices at maturity
    asset_prices = np.array([spot * (u**j) * (d ** (N - j)) for j in range(N + 1)])

    # Option values at maturity
    if option_type == "call":
        option_values = np.maximum(asset_prices - strike, 0)
    elif option_type == "put":
        option_values = np.maximum(strike - asset_prices, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    # Iterating backwards
    for i in range(N - 1, -1, -1):

        # This line of code continuously discounts the payoff based on the `nth`
        # iteration
        option_values = np.exp(-r * dt) * (
            p * option_values[1 : i + 2] + (1 - p) * option_values[0 : i + 1]
        )

    return option_values[0]


# Example usage:
if __name__ == "__main__":
    price = binomial_method(
        spot=100, strike=110, r=0.035, sigma=0.25, T=1.0, N=1000, option_type="call"
    )
    print(f"Option Price: {price}")

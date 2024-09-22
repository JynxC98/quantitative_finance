import numpy as np

def barrier_option(type=1, **params):
    """
    Monte-Carlo simulation code to simulate the knock-in options.

    Parameters:
    S0: Initial stock price
    K: Strike
    H: Barrier price
    T: Time to maturity
    r: Risk free rate
    sigma: Volatility
    N: Number of time steps
    M: Number of simulations
    type: The type of option
    1: Up-and-in Call
    0: Up-and-in Put
    -1: Down-and-in Call
    -2: Down-and-in Put
    """
    S0 = params.get("S0", 100)
    K = params.get("K", 100)
    H = params.get("H", 125)
    T = params.get("T", 1.0)
    r = params.get("r", 0.01)
    sigma = params.get("sigma", 0.2)
    N = params.get("N", 100)
    M = params.get("M", 10000)

    dt = T / N
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate M paths with N steps
    Z = np.random.standard_normal((M, N))
    S = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))

    # Add the initial stock price to the beginning of each path
    S = np.hstack((S0 * np.ones((M, 1)), S))

    # Check if the barrier is hit for each path
    if type in [1, 0]:  # Up-and-in
        barrier_hit = np.any(S >= H, axis=1)
    else:  # Down-and-in
        barrier_hit = np.any(S <= H, axis=1)

    # Calculate the payoff
    if type in [1, -1]:  # Call options
        payoff = np.maximum(S[:, -1] - K, 0)
    else:  # Put options
        payoff = np.maximum(K - S[:, -1], 0)

    # Apply the knock-in condition
    payoff = payoff * barrier_hit

    # Calculate the option price
    option_price = np.exp(-r * T) * np.mean(payoff)

    return option_price

if __name__ == "__main__":
    print("Up-and-in Call:", barrier_option(type=1))
    print("Up-and-in Put:", barrier_option(type=0))
    print("Down-and-in Call:", barrier_option(type=-1))
    print("Down-and-in Put:", barrier_option(type=-2))
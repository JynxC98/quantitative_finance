"""Implementation of Geometric Brownian Motion
"""

import numpy as np
import matplotlib.pyplot as plt


def brownian_motion(dt: float = 0.001, x0=0, n_iter=1000):
    """Simulates brownian motion."""

    # For weiner process, W(t) = 0
    W = np.zeros(n_iter + 1)
    t = np.linspace(x0, n_iter, n_iter + 1)

    # We have to use cumulative sum: On each step the additional value is
    # drawn from a normal distribution with mean 0 and variance dt N(0, dt)
    # Also, N(0, dt) = sqrt(dt)*N(0, 1)

    W[1 : n_iter + 1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n_iter))
    return t, W


def plot_process(t, W):
    """Plots the brownian motion"""

    plt.plot(t, W)
    plt.xlabel("Time(t)")
    plt.ylabel("Weiner-Process W(t)")
    plt.title("Weiner Process")
    plt.show()


if __name__ == "__main__":
    time, data = brownian_motion()
    plot_process(time, data)

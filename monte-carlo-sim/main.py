"""
Script to create a table for time taken for each simulation.
"""

import time
from collections import defaultdict
import pandas as pd


from models import simulate_heston_model_euler


def perform_simulations(step_sizes, strike_prices, maturity_times):
    """
    Main function to perform all the simulations and return the result into a
    `.csv` format.
    """

    required_data = defaultdict(list)
    for maturity_time in maturity_times:
        for strike_price in strike_prices:
            for step_size in step_sizes:
                start_time = time.time()
                result = simulate_heston_model_euler(
                    S_0=100,
                    v_0=0.09,
                    theta=0.348,
                    sigma=0.39,
                    kappa=1.15,
                    rho=-0.64,
                    risk_free_rate=0.05,
                    time_to_maturity=maturity_time,
                    strike_price=strike_price,
                    num_paths=50000,
                    step_size=step_size,
                )
                end_time = time.time()
                time_difference = end_time - start_time
                required_data["Time to Maturity"].append(maturity_time)
                required_data["Strike Price"].append(strike_price)
                required_data["Step Size"].append(step_size)
                required_data["Option Price"].append(
                    result.get("Mean Call Option Price")
                )
                required_data["Confidence Interval"].append(
                    result.get("Confidence Interval")
                )
                required_data["CPU"].append(time_difference)

    data = pd.DataFrame(required_data)
    data.to_csv("result_table.csv")


if __name__ == "__main__":
    STEP_SIZES = [10e-3, 10e-4, 10e-5]
    STRIKE_PRICES = [90, 100, 110]
    MATURITY_TIMES = [0.5, 1, 2]

    perform_simulations(
        step_sizes=STEP_SIZES,
        strike_prices=STRIKE_PRICES,
        maturity_times=MATURITY_TIMES,
    )

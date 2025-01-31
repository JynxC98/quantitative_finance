"""
This script compares the simulation prices with the analytical solution.
"""

from collections import defaultdict
import pandas as pd

from models import (
    calculate_option_price_bsm,
    antithetic_method,
    monte_carlo_sim,
    control_variate,
)


def simulate_data(functions, spot, strike, sigma, r, T, isCall=True):
    """
    The main script to simulate the data.

    Input parameters
    ---------------
    functions: The implemented functions of the pricing engines in the form {"name": function}.
    strike: The pre-determined strike price.
    sigma: Volatility
    r: Risk-free rate
    T: Time to maturity
    isCall: True for call and False for put
    """
    simulated_data = defaultdict(list)

    for method, function in functions.items():
        required_data = function(spot, strike, sigma, r, T)
        simulated_data["method"].append(method)
        simulated_data["mean_price"].append(required_data["mean_price"])
        simulated_data["upper_limit"].append(required_data["upper_limit"])
        simulated_data["lower_limit"].append(required_data["lower_limit"])
        simulated_data["std_dev"].append(required_data["std_dev"])

    return pd.DataFrame(simulated_data, index=range(0, len(functions)))


if __name__ == "__main__":

    NAMES = ["Antithetic", "Monte-Carlo", "Control-Variate"]
    SIMULATION_FUNCS = [antithetic_method, monte_carlo_sim, control_variate]

    required_dict = {name: func for (name, func) in zip(NAMES, SIMULATION_FUNCS)}

    SPOT = 100
    STRIKE = 120
    SIGMA = 0.3
    RATE = 0.045
    MAT = 1
    ISCALL = True

    analytical_value = calculate_option_price_bsm(
        spot=SPOT, strike=STRIKE, sigma=SIGMA, rf=RATE, T=MAT
    )
    print("*" * 50)
    print(f"The analytical value of option is {analytical_value}")
    print("*" * 50)
    print("Here are the simulation results")
    print("*" * 50)
    required_data = simulate_data(
        functions=required_dict, spot=SPOT, strike=STRIKE, sigma=SIGMA, r=RATE, T=MAT
    )
    print(required_data)
    print("*" * 50)

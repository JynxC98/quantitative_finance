"""
Comparative Analysis of Analytical Solutions and Monte Carlo Simulations for Option Pricing

This module demonstrates the differences between analytical solutions and Monte Carlo
simulations in the context of option pricing, specifically for geometric Asian options
under Heston's stochastic volatility model.

The code compares the efficiency and accuracy of the analytical solution implemented
in the 'analytical_solution' module against Monte Carlo simulations.

Key Features:
- Analytical pricing of geometric Asian options using the method from Kim and Wee (2011)
- Monte Carlo simulation of option prices 
- Performance comparison between analytical and simulation methods
- Accuracy analysis of both methods

References:
Kim, B., & Wee, I. S. (2011). Pricing of geometric Asian options under Heston's
stochastic volatility model. Quantitative Finance, 11(12), 1795-1811.
https://www.tandfonline.com/doi/abs/10.1080/14697688.2011.596844
"""

import warnings
from collections import defaultdict
import time
import pandas as pd
from analytical_solution import geometric_asian_call
from models import simulate_heston_model_milstein

warnings.filterwarnings("ignore")


def simulate(maturities, strikes, **kwargs):
    """
    Perform comparative simulations of analytical solutions and Monte Carlo methods.

    This function runs both the analytical solution and Monte Carlo simulations
    (to be implemented) for pricing geometric Asian options under various maturities
    and strike prices. It compares the results in terms of pricing accuracy and
    computational efficiency.

    Args:
        maturities (list of float): List of option maturities (in years) to simulate.
        strikes (list of float): List of strike prices to simulate.
        num_expansion (list of int): List of number of terms for expansion.
        **kwargs: Additional keyword arguments for the option pricing models.
            Expected arguments include:
            - S0 (float): Initial stock price
            - v0 (float): Initial volatility
            - theta (float): Long-term mean of volatility
            - sigma (float): Volatility of volatility
            - kappa (float): Mean reversion rate of volatility
            - rho (float): Correlation between stock price and volatility
            - r (float): Risk-free interest rate
            - n (int): Number of terms in series expansion for analytical solution

    Returns:
        dict: A dictionary containing the results of the simulations, including:
            - 'analytical_prices': List of prices from the analytical solution
            - 'monte_carlo_prices': List of prices from Monte Carlo simulation
            - 'analytical_time': Computation time for analytical solution
            - 'monte_carlo_time': Computation time for Monte Carlo simulation
            - 'price_differences': Differences between analytical and Monte Carlo prices
            - 'parameter_set': The set of parameters used for the simulations
    """
    result = defaultdict(list)
    S0 = kwargs.get("S0", 100)
    v0 = kwargs.get("v0", 0.09)
    r = kwargs.get("r", 0.05)
    theta = kwargs.get("theta", 0.348)
    rho = kwargs.get("rho", -0.64)
    kappa = kwargs.get("kappa", 1.15)
    sigma = kwargs.get("sigma", 0.39)
    n = kwargs.get("n", 30)

    for maturity in maturities:
        for strike in strikes:

            print(f"Evaluating maturity {maturity} and strike {strike}")

            result["maturity"].append(maturity)
            result["strike"].append(strike)
            start_time = time.time()
            option_price_analytic = geometric_asian_call(
                S0=S0,
                v0=v0,
                theta=theta,
                sigma=sigma,
                kappa=kappa,
                rho=rho,
                r=r,
                n=n,
                T=maturity,
                K=strike,
            )
            end_time = time.time()
            result["option_price_analytic"].append(option_price_analytic)

            total_time_analytic = end_time - start_time
            result["time_taken_analytic"].append(total_time_analytic)

            start_time = time.time()
            option_data = simulate_heston_model_milstein(
                S0=S0,
                v0=v0,
                theta=theta,
                sigma=sigma,
                kappa=kappa,
                rho=rho,
                r=r,
                n=n,
                T=maturity,
                K=strike,
            )
            end_time = time.time()

            result["option_price_simulation"].append(
                option_data["Mean Call Option Price"]
            )
            result["option_interval_simulation"].append(
                option_data["Confidence Interval"]
            )

            total_time_simulation = end_time - start_time
            result["time_taken_simulation"].append(total_time_simulation)
    required_result = pd.DataFrame(result)
    required_result.to_csv("simulation_result.csv", index=False)
    return required_result


if __name__ == "__main__":
    maturities_list = [0.2, 0.4]  # , 0.5, 1]
    strikes_list = [90, 95]  # , 100, 105, 110]

    simulation = simulate(maturities=maturities_list, strikes=strikes_list)
    print(simulation)

"""
Benchmark Script for the Mixed-Exponential Jump-Diffusion (MEJD) Model.

This module generates structured benchmark tables for European call option
prices computed under the Mixed-Exponential Jump-Diffusion model.

This layout mirrors the format commonly used in academic papers
(e.g., Cai & Kou, 2011).

Author: Harsh Parikh
"""

import numpy as np
import pandas as pd

from helpers import black_scholes_analytical
from mixed_exp_jd import calculate_price_med


def generate_benchmark_table(
    spot,
    strike,
    r,
    lambda_vals,
    sigmas,
    maturities,
    option_type,
    **params,
):
    """
    Generates a structured benchmark table for the MEJD model,
    including Black-Scholes comparison.

    Returns
    -------
    pd.DataFrame
        MultiIndex benchmark table structured as:

            Rows:    (Maturity, Lambda)
            Columns: (Sigma, Metric)

        Metrics:
            - MED value
            - BS value
    """

    records = []

    for T in maturities:
        for lam in lambda_vals:
            for sigma in sigmas:

                params["lambda_"] = lam

                med_price = calculate_price_med(
                    spot=spot,
                    strike=strike,
                    sigma=sigma,
                    r=r,
                    T=T,
                    option_type=option_type,
                    **params,
                )

                bs_price = black_scholes_analytical(
                    spot=spot,
                    strike=strike,
                    sigma=sigma,
                    r=r,
                    T=T,
                    option_type=option_type,
                )

                records.append(
                    {
                        "Maturity": T,
                        "Lambda": lam,
                        "Sigma": sigma,
                        "MED value": med_price,
                        "BS value": bs_price,
                    }
                )

    df = pd.DataFrame(records)

    table = df.pivot_table(
        index=["Maturity", "Lambda"],
        columns="Sigma",
        values=["MED value", "BS value"],
    )

    # Put sigma first in column grouping
    table = table.swaplevel(axis=1).sort_index(axis=1)

    table.index.names = ["Maturity", "Lambda"]
    table.columns.names = ["Sigma", "Metric"]

    return table.round(6)


def main():
    """
    Main execution block for benchmark generation.
    Adjust parameters here for experimentation.
    """

    # Core market parameters
    spot = 100.0
    strike = 100.0
    r = 0.05

    # Contract parameters
    option_type = "call"

    # Parameter grids
    lambda_vals = [0, 1, 3]
    sigmas = [0.2, 0.3, 0.5]
    maturities = [0.5, 1.0, 2.0]

    # Default MEJD parameters (can be modified)
    params = {
        "p_u": 0.4,
        "p_d": 0.6,
        "weights_up": [1.2, -0.2],
        "weights_down": [1.3, -0.3],
        "scaling_up": [20, 50],
        "scaling_down": [20, 50],
    }

    table = generate_benchmark_table(
        spot=spot,
        strike=strike,
        r=r,
        lambda_vals=lambda_vals,
        sigmas=sigmas,
        maturities=maturities,
        option_type=option_type,
        **params,
    )

    print(f"\nBenchmark Table for European {option_type} Options under MEJD\n")
    print(table)


if __name__ == "__main__":
    main()

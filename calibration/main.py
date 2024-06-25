"""
The main script for calibration.
"""

import numpy as np
from scipy.optimize import minimize

from models import heston_call_price
from fetch_data import get_strike_price_pivot_table


def calibrate_model(
    spot_price, strikes, maturities, call_option_prices, risk_free=0.04
):
    """
    Main calibration function.
    """
    params = {
        "v0": {"x0": 0.1, "lbub": [1e-3, 0.1]},
        "kappa": {"x0": 3, "lbub": [1e-3, 5]},
        "theta": {"x0": 0.05, "lbub": [1e-3, 0.1]},
        "sigma": {"x0": 0.3, "lbub": [1e-2, 1]},
        "rho": {"x0": -0.8, "lbub": [-1, 1]},
        "lambda_": {"x0": 0.03, "lbub": [-1, 1]},  # Changed "lambd" to "lambda_"
    }
    initial_values = [param["x0"] for _, param in params.items()]
    bounds = [param["lbub"] for _, param in params.items()]

    # Feller condition constraint
    def feller_constraint(x):
        _, kappa, theta, sigma, _, _ = x
        return 2 * kappa * theta - sigma**2

    constraints = [{"type": "ineq", "fun": feller_constraint}]

    def objective_function(x):
        v0, kappa, theta, sigma, rho, lambda_ = x
        total_error = 0.0
        for K, tau, market_price in zip(strikes, maturities, call_option_prices):
            model_price = heston_call_price(
                spot_price, K, v0, kappa, theta, sigma, rho, lambda_, risk_free, tau
            )
            # Using relative error
            total_error += np.sum(
                ((market_price - model_price) / (market_price)) ** 2
            )  # Added small constant to avoid division by zero
        return total_error

    result = minimize(
        objective_function,
        initial_values,
        method="L-BFGS-B",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-6},
    )

    if result.success:
        print("Optimization successful.")
    else:
        print("Optimization failed. Reason:", result.message)

    return result


if __name__ == "__main__":
    TICKER = "AAPL"
    price, required_table = get_strike_price_pivot_table(ticker=TICKER)
    required_strikes = required_table.columns.values
    required_maturities = required_table.index.values
    market_prices = required_table.values
    print(
        calibrate_model(
            spot_price=price,
            strikes=required_strikes,
            maturities=required_maturities,
            call_option_prices=market_prices,
        )
    )

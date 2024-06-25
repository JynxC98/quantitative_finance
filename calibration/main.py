"""
The main script for calibration.
"""

from scipy.optimize import minimize

from models import heston_call_price
from fetch_data import get_strike_price_pivot_table


def calibrate_model(
    spot_price, strikes, maturities, call_option_prices, risk_free=0.04
):
    """
    Calibrate the Heston model parameters to market data.

    This function performs the calibration of the Heston stochastic volatility model
    using market prices of European call options. The calibration minimizes the
    difference between the model prices and the market prices of the options.

    Parameters:
    -----------
    spot_price : float
        The current price of the underlying asset.
    strikes : list of float
        A list of strike prices for the options.
    maturities : list of float
        A list of times to maturity for the options, in years.
    call_option_prices : list of list of float
        A nested list where each inner list contains the market prices of call options
        corresponding to the given strike prices for a specific maturity.
    risk_free : float, optional (default=0.04)
        The risk-free interest rate, expressed as a decimal.

    Returns:
    --------
    result : OptimizeResult
        The result of the optimization process, containing the optimal parameters
        and information about the optimization process.

    Notes:
    ------
    The Heston model parameters to be calibrated are:
        v0 : Initial variance
        kappa : Rate of mean reversion
        theta : Long-term variance
        sigma : Volatility of volatility
        rho : Correlation between the two Brownian motions
        lambda_ : Volatility premium

    The optimisation is subject to the Feller condition constraint to ensure
    the variance remains positive.
    """
    params = {
        "v0": {"x0": 0.1, "lbub": [1e-3, 1]},
        "kappa": {"x0": 3, "lbub": [1e-3, 5]},
        "theta": {"x0": 0.05, "lbub": [1e-3, 1]},
        "sigma": {"x0": 0.3, "lbub": [1e-2, 1]},
        "rho": {"x0": -0.8, "lbub": [-1, 1]},
        "lambda_": {"x0": 0.03, "lbub": [-1, 1]},
    }
    initial_values = [param["x0"] for param in params.values()]
    bounds = [param["lbub"] for param in params.values()]

    # Feller condition constraint
    def feller_constraint(x):
        _, kappa, theta, sigma, _, _ = x
        return 2 * kappa * theta - sigma**2

    constraints = [{"type": "ineq", "fun": feller_constraint}]

    def objective_function(x):
        v0, kappa, theta, sigma, rho, lambda_ = x
        total_error = 0.0
        for i, tau in enumerate(maturities):
            for j, strike in enumerate(strikes):
                calculated_price = heston_call_price(
                    S0=spot_price,
                    K=strike,
                    v0=v0,
                    tau=tau,
                    kappa=kappa,
                    theta=theta,
                    sigma=sigma,
                    rho=rho,
                    lambda_=lambda_,
                    r=risk_free,
                )
                total_error += (calculated_price - call_option_prices[i][j]) ** 2
        return total_error

    result = minimize(
        objective_function,
        initial_values,
        method="SLSQP",
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

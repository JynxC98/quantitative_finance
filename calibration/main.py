"""
The main script for calibration.
"""

import time
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline

from models import heston_call_price
from fetch_data import get_strike_price_pivot_table


class CalibrateModel:
    """
    Calibrates the Heston stochastic volatility model under the risk-neutral pricing measure.

    This class implements the calibration of the Heston model parameters to match observed
    market option prices. The Heston model is a popular stochastic volatility model used
    in quantitative finance for option pricing and risk management.


    The Heston model parameters calibrated include:
    - v0: Initial variance
    - kappa: Mean reversion speed of variance
    - theta: Long-term mean of variance
    - sigma: Volatility of variance
    - rho: Correlation between stock price and variance processes
    - lambda: Price of volatility risk (optional, depending on implementation)

    Usage:
    This class is typically used in financial applications for:
    - Pricing exotic options
    - Calculating implied volatility surfaces
    - Risk management and scenario analysis
    - Studying the dynamics of asset price volatility

    Note:
    The calibration process can be computationally intensive and may require
    careful initialisation of parameters and choice of optimisation algorithm.
    """

    def __init__(self, ticker, yields, maturities):
        """
        This constructor sets up the `CalibrateModel` object for a given stock ticker,
        preparing it for subsequent data retrieval, processing, and model calibration.


        Input Parameters
        ----------------
            - ticker(str): stock ticker for a particular asset class.
            - yields(numpy.ndarray): Interest rates corresponding to different maturities.
            - maturities(numpy.ndarray): Maturities in years.
        """
        self.ticker = ticker

        self.option_data = get_strike_price_pivot_table(self.ticker)
        self.yields = yields
        self.maturities = maturities
        # Parameters to be calculated later
        self.spot_price_ = None
        self.parameter_history = []
        self.params_ = []

        # These parameters ensure that the methods are fitted properly.

        self._is_interpolated = False

    def interpolate_data(self, num_interpolations=20):
        """
        Interpolates the existing option data to create a denser grid for improved model accuracy.

        Parameters:
        -----------
        num_interpolations : int, optional (default=20)
            The number of interpolated points to add between each pair of existing data points.
            Higher values create a finer grid but increase computation time.

        """
        spot_price, pivot_table = self.option_data

        print("Successfully acquired the option data")
        self.spot_price_ = spot_price

        maturities = pivot_table.index.values
        strikes = pivot_table.columns.values
        option_prices = pivot_table.values

        curve_fit, _ = calibrate_nss_ols(self.maturities, self.yields)

        # Calibrating the interest rates.
        pivot_table["rate"] = pivot_table.index.map(curve_fit) * 0.01

        # Parameters to interpolate the option pricing data.
        time_grid = np.linspace(maturities.min(), maturities.max(), num_interpolations)
        strike_grid = np.linspace(strikes.min(), strikes.max(), num_interpolations)
        interpolator = RectBivariateSpline(maturities, strikes, option_prices)
        interpolated_prices = interpolator(time_grid, strike_grid)

        interpolated_int_rates = np.array([curve_fit(t) for t in time_grid]) * 0.01

        required_data = {
            "time_grid": time_grid,
            "strike_grid": strike_grid,
            "interpolated_prices": interpolated_prices,
            "interpolated_int_rates": interpolated_int_rates,
        }

        self._is_interpolated = True

        # To map the progress
        print("Successfully interpolated the option data")
        return required_data

    def calibrate_model(self):
        """
        Calibrate the Heston model using market data.

        This method performs the following steps:
        1. Retrieves the spot price and interpolates necessary market data.
        2. Sets up the optimization problem with initial parameter guesses and bounds.
        3. Defines the objective function to minimize the squared difference between
        model prices and market prices.
        4. Uses the L-BFGS-B algorithm to find the optimal model parameters.

        Returns:
            tuple: A tuple containing two elements:
                - result (OptimizeResult): The optimization result from scipy.optimize.minimize.
                - parameter_history (list): A list of parameter sets explored during optimization.

        Raises:
            Any exceptions raised by the optimization process are not explicitly handled.

        Note:
            This method assumes that necessary data (spot price, interpolated market data)
            is already available as instance attributes.
        """
        print("Calibrating model")
        spot_price = self.spot_price_
        required_parameters = self.interpolate_data()
        strikes = required_parameters["strike_grid"]
        maturities = required_parameters["time_grid"]
        option_prices = required_parameters["interpolated_prices"]
        risk_free = required_parameters["interpolated_int_rates"]

        params = {
            "v0": {"x0": 0.1, "bounds": (1e-2, 1)},
            "kappa": {"x0": 3, "bounds": (1e-2, 5)},
            "theta": {"x0": 0.05, "bounds": (1e-2, 1)},
            "sigma": {"x0": 0.3, "bounds": (1e-2, 2)},
            "rho": {"x0": -0.8, "bounds": (-0.99, 0.99)},
            "lambda_": {"x0": 0.03, "bounds": (-1, 1)},
        }
        initial_values = np.array([param["x0"] for param in params.values()])
        bounds = [param["bounds"] for param in params.values()]

        parameter_history = []

        def callback(x):
            """
            Callback function to record parameter history during optimization.

            Args:
                x (np.array): Current parameter set being evaluated.
            """
            parameter_history.append(x.copy())

        def fellers_constraint(x):
            """
            Feller's contstraint ensures non-negative variance.
            """
            _, kappa, theta, sigma, _, _ = x
            return 2 * kappa * theta - sigma**2

        def objective_function(x):
            """
            Objective function to be minimized in the calibration process.

            This function calculates the sum of squared differences between
            model prices and market prices, with a small regularization term.

            Args:
                x (np.array): Array of model parameters [v0, kappa, theta, sigma, rho, lambda_].

            Returns:
                float: The value of the objective function to be minimized.
            """
            v0, kappa, theta, sigma, rho, lambda_ = x
            model_prices = np.array(
                [
                    [
                        heston_call_price(
                            spot_price,
                            strike,
                            v0,
                            kappa,
                            theta,
                            sigma,
                            rho,
                            lambda_,
                            risk_free[i],
                            tau,
                        )
                        for strike in strikes
                    ]
                    for i, tau in enumerate(maturities)
                ]
            )

            return np.sum((model_prices - option_prices) ** 2) + 1e-6 * np.sum(x**2)

        result = minimize(
            objective_function,
            initial_values,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-4, "maxiter": 1000},
            constraints=fellers_constraint,
            callback=callback,
        )

        if result.success:
            print("Optimisation successful.")
        else:
            print("Optimisation failed. Reason:", result.message)

        return result, parameter_history


if __name__ == "__main__":
    TICKER = "AAPL"
    # Fed rates as of July 2024.
    YIELDS = np.array(
        [1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    )
    MATURITIES = np.array(
        [5.47, 5.48, 5.52, 5.46, 5.40, 5.16, 4.87, 4.62, 4.48, 4.47, 4.47, 4.68, 4.59]
    )

    model = CalibrateModel(ticker=TICKER, yields=YIELDS, maturities=MATURITIES)
    start = time.time()
    required_result, _ = model.calibrate_model()
    end = time.time()
    print(required_result.x)
    print(f"Time taken = {end - start}")

"""
The main script for calibration.
"""

import time
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

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
        self.yields = yields
        self.maturities = maturities
        # Parameters to be calculated later
        self.spot_price_ = None
        self.parameter_history = []
        self.params_ = []
        self.data = {}

    def get_data(self):
        """
        Generates the required data for calibration.
        """
        spot_price, pivot_table = get_strike_price_pivot_table(self.ticker)

        print("Successfully acquired the option data")
        self.spot_price_ = spot_price

        maturities = pivot_table.index.values
        strikes = pivot_table.columns.values
        option_prices = pivot_table.values

        curve_fit, _ = calibrate_nss_ols(self.maturities, self.yields)

        # Calibrating the interest rates.
        pivot_table["rate"] = pivot_table.index.map(curve_fit) * 0.01
        rates = pivot_table["rate"].values

        required_data = {
            "strikes": strikes,
            "maturities": maturities,
            "option_prices": option_prices,
            "rates": rates,
        }
        return required_data

    def calibrate_model(self):
        """
        Calibrate the Heston model using market data.

        This method performs the following steps:
        1. Retrieves the spot price and interpolates necessary market data.
        2. Sets up the optimization problem with initial parameter guesses and bounds.
        3. Defines the objective function to minimize the squared difference between
        model prices and market prices.
        4. Uses the SLSQP algorithm to find the optimal model parameters.

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
        required_parameters = self.get_data()
        spot_price = self.spot_price_
        strikes = required_parameters["strikes"]
        maturities = required_parameters["maturities"]
        option_prices = required_parameters["option_prices"]
        risk_free = required_parameters["rates"]
        self.data["spot_price"] = spot_price
        self.data["strikes"] = strikes
        self.data["maturities"] = maturities
        self.data["option_prices"] = option_prices
        self.data["risk_free"] = risk_free

        params = {
            "v0": {"x0": 0.1, "bounds": (1e-2, 1)},
            "kappa": {"x0": 3, "bounds": (1e-2, 5)},
            "theta": {"x0": 0.05, "bounds": (1e-2, 1)},
            "sigma": {"x0": 0.3, "bounds": (1e-2, 2)},
            "rho": {"x0": -0.8, "bounds": (-1, 1)},
            "lambda_": {"x0": 0.03, "bounds": (-1, 1)},
        }
        initial_values = np.array([param["x0"] for param in params.values()])
        bounds = [param["bounds"] for param in params.values()]

        def callback(x):
            """
            Callback function to record parameter history during optimization.

            Args:
                x (np.array): Current parameter set being evaluated.
            """
            self.parameter_history.append(x.copy())

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
            v0, kappa, theta, sigma, rho, lambda_ = x
            model_prices = np.array([[heston_call_price(spot_price, strike, v0,
                                                        kappa, theta, sigma, rho, lambda_, risk_free[i], tau)
                                    for strike in strikes] for i, tau in enumerate(maturities)])

            mse = np.sum((model_prices - option_prices)**2) / (len(strikes) * len(maturities))

            # Adding a penalty for violating Feller's condition
            feller_violation = max(0, sigma**2 - 2 * kappa * theta)
            penalty = 1e6 * feller_violation  # Large penalty for violation
            return mse + penalty + 1e-4 * np.sum(x**2)

        fellers_constraint_dict = {"type": "ineq", "fun": fellers_constraint}

        result = minimize(objective_function, initial_values, method='SLSQP',
                    bounds=bounds, constraints=[fellers_constraint_dict],
                    options={'ftol': 1e-4, 'maxiter': 1000}, callback=callback)

        if result.success:
            print("Optimisation successful.")
        else:
            print("Optimisation failed. Reason:", result.message)

        return result
    
    def plot_calibration_result(self):
        """
        Plots the convergence of the optimisation function.
        """
    # Check if the model has been calibrated
        if not hasattr(self, 'parameter_history') or self.parameter_history is None:
            raise RuntimeError("Model calibration has not been performed yet. Please run 'calibrate_model' first.")
        
        param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho', 'lambda_']
        param_history = np.array(self.parameter_history)

        # Set the style and color palette
        sns.set(style="whitegrid", palette="deep")

        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('Heston Model Parameter Convergence', fontsize=20, y=1.02)

        for i, (ax, name) in enumerate(zip(axs.flatten(), param_names)):
            sns.lineplot(data=param_history[:, i], ax=ax, linewidth=2, marker='o')
            ax.set_title(f'{name.capitalize()} Convergence', fontsize=14)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Parameter Value', fontsize=12)
            ax.tick_params(labelsize=10)

            # Add final value annotation
            final_value = param_history[-1, i]
            ax.annotate(f'Final: {final_value:.4f}',
                        xy=(len(param_history)-1, final_value),
                        xytext=(0.7, 0.95),
                        textcoords='axes fraction',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))

        plt.tight_layout()
        fig.subplots_adjust(top=0.93)  # Adjust to prevent title overlap

        # Add a text box with calibration summary
        convergence_text = (f"Calibration converged in {len(self.parameter_history)} iterations.\n"
                            f"Final parameter values:\n"
                            f"v0: {param_history[-1, 0]:.4f}\n"
                            f"kappa: {param_history[-1, 1]:.4f}\n"
                            f"theta: {param_history[-1, 2]:.4f}\n"
                            f"sigma: {param_history[-1, 3]:.4f}\n"
                            f"rho: {param_history[-1, 4]:.4f}\n"
                            f"lambda: {param_history[-1, 5]:.4f}")

        fig.text(0.5, -0.05, convergence_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

        plt.savefig('heston_parameter_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()





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
    required_result = model.calibrate_model()
    end = time.time()
    print(required_result.x)
    print(f"Time taken = {end - start}")

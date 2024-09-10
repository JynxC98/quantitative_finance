"""
Calibration of Heston model using Quantlib.
Refer this site for documentation: https://www.quantlib.org/docs.shtml
"""

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import matplotlib.pyplot as plt
import seaborn as sns
import QuantLib as ql

from option_data_and_calculations import (
    get_strike_price_pivot_table,
    bsm_implied_volatility,
)

warnings.filterwarnings("ignore")


class QuantlibCalibration:
    """
    A class for calibrating the Heston model using Quantlib.

    This class handles the entire process of calibrating the Heston model,
    from data preprocessing to model fitting and visualization of the
    volatility surface.

    Attributes:
        ticker (str): The stock ticker symbol.
        yields (np.array): Array of yield values.
        yield_maturities (np.array): Array of maturities corresponding to the yields.
        option_type (str): Type of option, either "call" or "put".
        params (tuple): Calibrated Heston model parameters.
        data (dict): Preprocessed market data.
        model (ql.HestonModel): Calibrated Heston model.
        calculation_date (ql.Date): The date of calculation.
    """

    def __init__(self, ticker, yields, yield_maturities, option_type="call"):
        """
        Initialize the QuantlibCalibration object.

        Args:
            ticker (str): The stock ticker symbol.
            yields (np.array): Array of yield values.
            yield_maturities (np.array): Array of maturities corresponding to the yields.
            option_type (str, optional): Type of option. Defaults to "call".
        """
        self.ticker = ticker
        self.yields = yields
        self.yield_maturities = yield_maturities
        self.option_type = option_type

        # These values will be calculated later.
        self.params = None
        self.data = None
        self.model = None
        self.calculation_date = None

    def data_preprocessing(self):
        """
        Preprocess the market data for Heston model calibration.

        This method retrieves option data, calculates implied volatilities,
        and prepares the data structure for the calibration process.
        """
        required_data = get_strike_price_pivot_table(self.ticker)
        pivot_table = required_data["Pivot Table"]
        spot_price = required_data["spot_price"]
        valid_maturities = required_data["Valid Maturities"]

        maturities = pivot_table.index.values
        strikes = pivot_table.columns.values
        option_prices = pivot_table.values

        # Second argument below gives us the statistics of the curve fit, not required 
        # for the current implementation. 
    
        curve_fit, _ = calibrate_nss_ols(self.yield_maturities, self.yields)

        # Converting the percentage values to absolute values
        pivot_table["rate"] = pivot_table.index.map(curve_fit) * 0.01
        rates = pivot_table["rate"].values

        imp_vols = np.zeros([len(valid_maturities), len(strikes)])
        for i, tau in enumerate(maturities):
            for j, strike in enumerate(strikes):
                imp_vols[i, j] = bsm_implied_volatility(
                    spot_price, strike, rates[i], tau, option_prices[i, j]
                )

        # The expiration date is in the form of YYYY-MM-DD
        expiration_dates = [
            ql.Date(
                int(date.split("-")[2]), # Extracting year
                int(date.split("-")[1]), # Extracting month
                int(date.split("-")[0]), # Extracting day
            )
            for date in valid_maturities
        ]

        self.data = {
            "strikes": strikes,
            "maturities": maturities,
            "spot_price": spot_price,
            "expiration_dates": expiration_dates,
            "imp_vols": imp_vols,
            "rates": rates,
        }

    def fit_model(self):
        """
        Fit the Heston model to the preprocessed market data.

        This method calibrates the Heston model using the Levenberg-Marquardt
        algorithm and stores the calibrated model and its parameters.
        """
        _ = self.data_preprocessing()
        data = self.data
        strikes = data["strikes"]
        maturities = data["maturities"]
        rates = data["rates"]
        imp_vols = data["imp_vols"]
        expiration_dates = data["expiration_dates"]
        spot_price = data["spot_price"]

        day_count = (
            ql.Actual365Fixed()
        )  # This line ensures that the number of days are set to 365
        calendar = ql.UnitedStates(
            ql.UnitedStates.GovernmentBond
        )  # Calendar set according to the American standard

        today = datetime.today()
        calculation_date = ql.Date(today.day, today.month, today.year)
        self.calculation_date = calculation_date

        ql.Settings.instance().evaluationDate = (
            calculation_date  # This line is used to set a pivot for today's date.
        )

        # Converting the `yield_maturities` into `Quantlib.Date` object.
        dates = [
            calculation_date + ql.Period(int(maturity * 12), ql.Months)
            for maturity in self.yield_maturities
        ]

        # Calibrating the interest rates with respect to the yield maturities.
        zero_curve = ql.ZeroCurve(dates, rates, day_count, calendar)
        zero_curve_handle = ql.YieldTermStructureHandle(zero_curve)

        # Calculating dividend yield.
        dividend_rate = 0.0
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, dividend_rate, day_count)
        )

        imp_vols_quantlib = ql.Matrix(len(strikes), len(maturities))
        for i, _ in enumerate(strikes):
            for j, _ in enumerate(maturities):
                # `imp_vols` has implied volatilities for different strikes per maturity.
                # According to the `Quantlib` documentation, the `Quantlib` matrix should
                # have strikes as rows and maturities as columns
                imp_vols_quantlib[i][j] = imp_vols[j, i]

        # Interpolating the surface across maturities and strikes.
        black_var_surface = ql.BlackVarianceSurface(
            calculation_date,
            calendar,
            expiration_dates,
            strikes,
            imp_vols_quantlib,
            day_count,
        )

        # Initial guess for the Heston parameters.
        v0, kappa, theta, rho, sigma = 0.01, 0.2, 0.02, -0.75, 0.5
        process = ql.HestonProcess(
            zero_curve_handle,
            dividend_ts,
            ql.QuoteHandle(ql.SimpleQuote(spot_price)),
            v0,
            kappa,
            theta,
            sigma,
            rho,
        )

        model = ql.HestonModel(process)
        self.model = model
        engine = ql.AnalyticHestonEngine(model)
        heston_helpers = []
        black_var_surface.setInterpolation("bicubic")

        for i, date in enumerate(expiration_dates):
            for j, strike in enumerate(strikes):
                t = date - calculation_date
                p = ql.Period(t, ql.Days)
                sigma = imp_vols[i][j]
                helper = ql.HestonModelHelper(
                    p,
                    calendar,
                    spot_price,
                    strike,
                    ql.QuoteHandle(ql.SimpleQuote(sigma)),
                    zero_curve_handle,
                    dividend_ts,
                )
                helper.setPricingEngine(engine)
                heston_helpers.append(helper)

        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        model.calibrate(
            heston_helpers, lm, ql.EndCriteria(500, 50, 1.0e-8, 1.0e-8, 1.0e-8)
        )

        self.params = model.params()

    def generate_heston_surface(self, num_strikes=50, num_maturities=20):
        """
        Generate the Heston model implied volatility surface.

        Args:
            num_strikes (int, optional): Number of strike prices. Defaults to 50.
            num_maturities (int, optional): Number of maturities. Defaults to 20.

        Returns:
            tuple: A tuple containing:
                - np.array: Array of strike prices.
                - np.array: Array of maturities.
                - np.array: 2D array representing the implied volatility surface.

        Raises:
            ValueError: If the model has not been calibrated.
        """
        if self.model is None:
            raise ValueError("Model has not been calibrated. Call fit_model() first.")

        data = self.data
        spot_price = data["spot_price"]
        min_strike = min(data["strikes"])
        max_strike = max(data["strikes"])
        min_maturity = min(data["maturities"])
        max_maturity = max(data["maturities"])

        strikes = np.linspace(min_strike, max_strike, num_strikes)
        maturities = np.linspace(min_maturity, max_maturity, num_maturities)

        heston_surface = np.zeros((num_maturities, num_strikes))

        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

        for i, maturity in enumerate(maturities):
            for j, strike in enumerate(strikes):
                # Used to calculate the number of days until maturity.
                expiry_date = self.calculation_date + ql.Period(
                    int(maturity * 365), ql.Days
                )

                option_type = (
                    ql.Option.Call if self.option_type == "call" else ql.Option.Put
                )

                # This function is used to simulate the equation: Payoff = E[S - K]^+
                payoff = ql.PlainVanillaPayoff(option_type, strike)
                exercise = ql.EuropeanExercise(expiry_date)
                option = ql.VanillaOption(payoff, exercise)

                # heston_process = self.model.process()
                option.setPricingEngine(ql.AnalyticHestonEngine(self.model))

                # The following lines of code ensure that the volatility surface is generated without any `None` values.
                try:
                    implied_vol = option.impliedVolatility(
                        option.NPV(),
                        ql.GeneralizedBlackScholesProcess(
                            ql.QuoteHandle(ql.SimpleQuote(spot_price)),
                            ql.YieldTermStructureHandle(
                                ql.FlatForward(self.calculation_date, 0.0, day_count)
                            ),
                            ql.YieldTermStructureHandle(
                                ql.FlatForward(self.calculation_date, 0.0, day_count)
                            ),
                            ql.BlackVolTermStructureHandle(
                                ql.BlackConstantVol(
                                    self.calculation_date, calendar, 0.1, day_count
                                )
                            ),
                        ),
                        1e-6,
                        100,
                        1e-7,
                        4.0,
                    )
                    heston_surface[i, j] = implied_vol
                except RuntimeError:
                    heston_surface[i, j] = np.nan

        return strikes, maturities, heston_surface

    def plot_volatility_surface(self, strikes, maturities, surface):
        """
        Plot the Heston model implied volatility surface.

        Args:
            strikes (np.array): Array of strike prices.
            maturities (np.array): Array of maturities.
            surface (np.array): 2D array representing the implied volatility surface.
        """
        X, Y = np.meshgrid(np.log(strikes / self.data["spot_price"]), maturities)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(X, Y, surface, cmap="viridis", edgecolor="none")

        ax.set_xlabel("Moneyness (Log Scale)")
        ax.set_ylabel("Maturity")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("Heston Model Implied Volatility Surface")

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(f"Heston volatility surface for {self.ticker}", dpi=500)
        plt.show()

    def calibrate_and_plot(self):
        """
        Calibrate the Heston model and plot the implied volatility surface.

        This method combines the model fitting, surface generation, and plotting
        steps into a single convenient function.
        """
        self.fit_model()
        strikes, maturities, surface = self.generate_heston_surface()
        self.plot_volatility_surface(strikes, maturities, surface)


if __name__ == "__main__":
    YIELDS = np.array(
        [5.47, 5.48, 5.52, 5.46, 5.40, 5.16, 4.87, 4.62, 4.48, 4.47, 4.47, 4.68, 4.59]
    )
    YIELD_MATURITIES = np.array(
        [1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    )
    TICKER = "AAPL"
    model = QuantlibCalibration(
        ticker=TICKER, yields=YIELDS, yield_maturities=YIELD_MATURITIES
    )

    model.fit_model()

    print(model.calibrate_and_plot())

    # Checking Feller's condition
    _, kappa, theta, _, sigma = model.params

    condition = 2 * kappa * theta / (sigma**2)

    if condition > 1:
        print("Feller's condition satisfied")
    else:
        print("Feller's condition not satisfied")

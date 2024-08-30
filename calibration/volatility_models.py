"""
Scripts related to calculating implied volatility.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from models import heston_call_price


def calculate_option_price(
    spot_price,
    strike_price,
    time_to_maturity,
    risk_free_rate,
    volatility,
    option_type="call",
):
    """
    Calculate the price of a European option using the Black-Scholes-Merton model.

    Parameters:
    -----------
    initial_price : float
        The current price of the underlying asset.
    strike_price : float
        The strike price of the option.
    time_to_maturity : float
        Time to maturity in years.
    risk_free_rate : float
        The risk-free interest rate (annualized).
    volatility : float
        The volatility of the underlying asset (annualized).
    option_type : str, optional
        The type of option, either "call" or "put" (default is "call").

    Returns:
    --------
    float
        The calculated option price.
    """

    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    d1 = (
        np.log(spot_price / strike_price)
        + (risk_free_rate + volatility**2 / 2) * time_to_maturity
    ) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    if option_type == "call":
        return spot_price * norm.cdf(d1) - strike_price * np.exp(
            -risk_free_rate * time_to_maturity
        ) * norm.cdf(d2)
    else:
        return strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(
            -d2
        ) - spot_price * norm.cdf(-d1)


def calculate_implied_volatility(spot, strike, risk_free, tau, option_price, v0, kappa, theta, sigma, rho, lambda_):
    """
    Calculate the implied volatility for a given option price using the Heston model.

    This function uses the Brent's method to find the implied volatility that makes the
    Black-Scholes option price equal to the Heston model option price.

    Parameters:
    -----------
    spot : float
        The current price of the underlying asset.
    strike : float
        The strike price of the option.
    risk_free : float
        The risk-free interest rate.
    tau : float
        Time to expiration in years.
    option_price : float
        The market price of the option.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-term mean of variance.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation between asset returns and variance.
    lambda_ : float
        Market price of volatility risk.

    Returns:
    --------
    float
        The implied volatility that equates the Black-Scholes price to the Heston price.

    Raises:
    -------
    ValueError
        If the root-finding algorithm fails to converge within the specified bounds.

    Notes:
    ------
    This function assumes the existence of two other functions:
    - calculate_option_price: Calculates the Black-Scholes option price.
    - heston_call_price: Calculates the Heston model option price.
    These functions should be defined elsewhere in the code.
    """
    low, high = 1e-4, 5.0
    
    def objective_function(imp_vol):
        """
        Define the objective function for root-finding.

        This function calculates the difference between the Black-Scholes price
        and the Heston model price for a given implied volatility.

        Parameters:
        -----------
        imp_vol : float
            The implied volatility to test.

        Returns:
        --------
        float
            The difference between Black-Scholes and Heston prices.
        """
        return (calculate_option_price(spot, strike, tau, risk_free, imp_vol) -
                heston_call_price(spot, strike, v0, kappa, theta, sigma, rho, lambda_, risk_free, tau))

    return brentq(lambda x: objective_function(x), a=low, b=high)
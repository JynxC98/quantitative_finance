"""
A script to calculate local volatility function.
"""

from datetime import datetime
import numpy as np
from data import get_option_data_from_yahoo

# Constants
NUM_DAYS = 252  # Number of trading days in one year
RISK_FREE_RATE = 0.05  # Risk-free rate


def calculate_local_volatility(option_data, expiration_date, **params):
    """
    Calculates local volatility based on Dupire's local volatility function.
    Uses finite difference approximation to calculate the local volatility.

    Parameters
    ----------
    option_data : dict or DataFrame
        Dictionary or DataFrame containing option prices and strike prices.
        It should have keys/columns for "ask" (option prices) and "strike" (strike prices).
    expiration_date : str
        Expiration date of the option in the format "YYYY-MM-DD".
    params : dict, optional
        Additional parameters.

    Returns
    -------
    local_volatility : np.ndarray
        Local volatility grid.

    """
    # Extract option prices and strike prices from option_data
    strike_prices = np.array(option_data["strike"])
    option_prices = np.array(option_data["ask"])

    # Calculate time to maturity
    today = datetime.today()
    expiration_datetime = datetime.strptime(expiration_date, "%Y-%m-%d")
    days_to_expiration = (expiration_datetime - today).days
    time_to_maturity = days_to_expiration / NUM_DAYS

    # Compute time and strike step sizes
    num_time_grid = params.get("N", 1000)
    spatial_grid = params.get("M", 1000)
    dt = time_to_maturity / num_time_grid
    dK = (max(strike_prices) - min(strike_prices)) / spatial_grid

    # Computing first order derivatives

    dC_dt = np.gradient(option_prices, dt)
    dC_dK = np.gradient(option_prices, dK)

    # Computing second order derivatives
    d2C_dK2 = np.gradient(dC_dK, dK)

    # Compute numerator and denominator for local volatility
    numerator = 2 * (dC_dt + RISK_FREE_RATE * strike_prices * dC_dK)
    denominator = (strike_prices**2) * d2C_dK2

    # Compute local volatility
    local_volatility = np.sqrt(numerator / denominator)

    return local_volatility


if __name__ == "__main__":
    # Example usage
    TICKER = "AAPL"
    EXPIRATION_DATE = "2025-03-21"
    req_option_data = get_option_data_from_yahoo(TICKER, EXPIRATION_DATE)

    result = calculate_local_volatility(req_option_data, EXPIRATION_DATE)
    print(result)

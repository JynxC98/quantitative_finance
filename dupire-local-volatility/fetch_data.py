import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf


def get_maturities(ticker):
    """
    Fetches the available maturity dates for a particular ticker.
    """
    company_data = yf.Ticker(ticker)
    expiration_dates = company_data.options

    return expiration_dates


def get_strike_price_pivot_table(
    ticker, maturities, spot_price, option_type="call", min_rows=30
):
    """
    Returns a pivot table with time to maturity as rows, strike prices as columns,
    and option prices as values.

    Parameters:
    - ticker: str, the ticker symbol of the company.
    - maturities: list of str, list of option maturities in 'YYYY-MM-DD' format.
    - spot_price: float, current price of the underlying asset.
    - option_type: str, "call" for call options, "put" for put options.
    - min_rows: int, minimum number of rows required to consider the option chain.

    Returns:
    - pivot_table: DataFrame, pivot table with TTM as rows, strike prices as columns, and option prices as values.
    """
    company_data = yf.Ticker(ticker)
    strikes = set(company_data.option_chain(maturities[0]).calls["strike"])
    options_data = []

    for maturity in maturities:
        data = company_data.option_chain(maturity)
        if option_type == "call":
            data = data.calls
        else:
            data = data.puts

        if data.shape[0] < min_rows:
            continue

        strikes = strikes.intersection(data["strike"])

        # Filter strikes within 95% to 110% of the spot price
        filtered_strikes = {
            strike
            for strike in strikes
            if 0.95 * spot_price <= strike <= 1.1 * spot_price
        }
        filtered_data = data[data["strike"].isin(filtered_strikes)]

        filtered_data["TTM"] = (
            pd.to_datetime(maturity) - pd.Timestamp.today()
        ).days / 252
        options_data.append(filtered_data)

    combined_data = pd.concat(options_data, ignore_index=True)
    pivot_table = combined_data.pivot_table(
        index="TTM", columns="strike", values="lastPrice", fill_value=0
    )

    return pivot_table


def calculate_option_price(spot, strike, sigma, risk_free, maturity, option_type=1):
    """
    Calculates the value of an option based on the Black-Scholes pricing model.

    Parameters:
    - spot: float, current price of the underlying asset
    - strike: float, strike price of the option
    - sigma: float, volatility of the underlying asset
    - risk_free: float, risk-free interest rate
    - maturity: float, time to maturity (in years)
    - option_type: int, 1 for Call option, 0 for Put option

    Returns:
    - float, price of the option
    """
    if option_type not in (0, 1):
        raise ValueError("Invalid option_type, please select 0 (Put) or 1 (Call)")

    d1 = (np.log(spot / strike) + (risk_free + 0.5 * sigma**2) * maturity) / (
        sigma * np.sqrt(maturity)
    )
    d2 = d1 - sigma * np.sqrt(maturity)

    if option_type == 1:
        return spot * norm.cdf(d1) - strike * np.exp(-risk_free * maturity) * norm.cdf(
            d2
        )
    else:
        return strike * np.exp(-risk_free * maturity) * norm.cdf(-d2) - spot * norm.cdf(
            -d1
        )


def implied_volatility(spot, strike, risk_free, maturity, actual_price, option_type=1):
    """
    Calculates the implied volatility of an option given the market price.

    Parameters:
    - spot: float, current price of the underlying asset
    - strike: float, strike price of the option
    - risk_free: float, risk-free interest rate
    - maturity: float, time to maturity (in years)
    - actual_price: float, market price of the option
    - option_type: int, 1 for Call option, 0 for Put option

    Returns:
    - float, implied volatility of the option
    """
    if actual_price == 0:
        return 0.0

    def objective_function(sigma):
        """
        Objective function to calculate implied volatility.

        Parameters:
        - sigma: float, volatility of the option

        Returns:
        - float, difference between calculated and actual option price
        """
        try:
            return (
                calculate_option_price(
                    spot, strike, sigma, risk_free, maturity, option_type=option_type
                )
                - actual_price
            )
        except Exception as e:
            raise e

    low, high = 1e-4, 5.0
    try:
        return brentq(lambda x: objective_function(x), low, high, xtol=1e-6)
    except Exception as e:
        print(
            f"Error for strike={strike}, maturity={maturity}, actual_price={actual_price}: {e}"
        )
        return np.nan

"""
A script to fetch the option data using yahoo finance API
"""

from collections import defaultdict
import warnings
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get_strike_price_pivot_table(
    ticker, maturity_min=0.1, maturity_max=2, moneyness_min=0.95, moneyness_max=1.2
):
    """
    Generate a pivot table of option strike prices for a given ticker.

    This function fetches option data for a specified stock ticker and creates a pivot table
    of strike prices across different maturities. It filters the data based on time to maturity
    and moneyness (strike price relative to spot price).

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
        maturity_min (float, optional): Minimum time to maturity in years. Defaults to 0.1.
        maturity_max (float, optional): Maximum time to maturity in years. Defaults to 2.
        moneyness_min (float, optional): Minimum moneyness (strike/spot) to consider. Defaults to 0.95.
        moneyness_max (float, optional): Maximum moneyness (strike/spot) to consider. Defaults to 1.2.

    Returns:
        pd.DataFrame: A pivot table with time to maturity as index, strike prices as columns,
                      and option last prices as values. Strikes not available for a particular
                      maturity are filled with 0.
    """
    option_data = yf.Ticker(ticker)
    spot_price = option_data.history("1D")["Close"].iloc[-1]

    today = pd.Timestamp.today()
    valid_maturities = [
        mat
        for mat in option_data.options
        if maturity_min < (pd.to_datetime(mat) - today).days / 252 < maturity_max
    ]

    strikes_freq = defaultdict(int)
    all_data = []

    for maturity in valid_maturities:
        chain = option_data.option_chain(maturity).calls
        ttm = (pd.to_datetime(maturity) - today).days / 252

        valid_strikes = chain[
            (chain["strike"] >= moneyness_min * spot_price)
            & (chain["strike"] <= moneyness_max * spot_price)
        ]

        for strike in valid_strikes["strike"]:
            strikes_freq[strike] += 1

        valid_strikes["TTM"] = ttm
        all_data.append(valid_strikes[["strike", "lastPrice", "TTM"]])

    common_strikes = {
        strike for strike, freq in strikes_freq.items() if freq == len(valid_maturities)
    }

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data[combined_data["strike"].isin(common_strikes)]

    pivot_table = combined_data.pivot_table(
        index="TTM", columns="strike", values="lastPrice", fill_value=0
    )

    # Fed rates as of July 2024.
    yield_maturities = np.array(
        [1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    )
    yields = np.array(
        [5.47, 5.48, 5.52, 5.46, 5.40, 5.16, 4.87, 4.62, 4.48, 4.47, 4.47, 4.68, 4.59]
    )

    # Calibrating interest rates
    curve_fit, _ = calibrate_nss_ols(yield_maturities, yields)

    pivot_table["rate"] = pivot_table.index.map(curve_fit) * 0.01
    return spot_price, pivot_table


if __name__ == "__main__":

    # Test usecase
    TICKER = "AAPL"
    print(get_strike_price_pivot_table(TICKER))

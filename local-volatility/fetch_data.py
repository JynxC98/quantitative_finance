"""
A script to generate a pivot table of strike prices across different maturities.
"""

from collections import defaultdict
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get_strike_price_pivot_table(
    ticker,
    maturity_min=0.1,
    maturity_max=2,
    moneyness_min=0.95,
    moneyness_max=1.2,
    option_type="call",
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
        option_type (str, optional): The type of option ('call' or 'put'). Defaults to 'call'.

    Returns:
        pd.DataFrame: A pivot table with time to maturity as index, strike prices as columns,
                    and option last prices as values. Strikes not available for a particular
                    maturity are filled with 0.
    """
    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    option_data = yf.Ticker(ticker)
    spot_price = option_data.history("1D")["Close"].iloc[0]

    today = pd.Timestamp.today()
    valid_maturities = [
        mat
        for mat in option_data.options
        if maturity_min < (pd.to_datetime(mat) - today).days / 365 < maturity_max
    ]

    strikes_freq = defaultdict(int)
    all_data = []

    for maturity in valid_maturities:
        if option_type == "call":
            chain = option_data.option_chain(maturity).calls
        elif option_type == "put":
            chain = option_data.option_chain(maturity).puts
        ttm = (pd.to_datetime(maturity) - today).days / 365

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

    return spot_price, pivot_table


if __name__ == "__main__":
    TICKER = "AAPL"
    table = get_strike_price_pivot_table(TICKER)
    print(table)

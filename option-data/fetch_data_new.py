"""
A script to generate a pivot table of implied volatility data across different maturities.
"""

import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")


def get_option_data(
    ticker,
    maturity_min=0.1,
    maturity_max=2,
    moneyness_min=0.85,
    moneyness_max=1.15,
    option_type="call",
    max_spread_ratio=3.0,
    min_volume=10,
):
    """
    Generate a pivot table of total implied variance for a given ticker.

    This function fetches option data for a specified stock ticker and creates a pivot table
    of total implied variance (sigma^2 * T) across different maturities and strikes.
    It applies several data quality filters to ensure only liquid, reliable options are used.

    Filters applied:
        - Zero bid removal: Options with zero bid price are illiquid and removed.
        - Spread ratio filter: Options where ask/bid exceeds max_spread_ratio are removed.
        - Volume filter: Options with insufficient trading volume are removed.
        - Moneyness filter: Only near-the-money options are retained for numerical stability.
        - Implied vol sanity check: Removes Yahoo Finance IV values outside [0.05, 2.0].

    Note:
        Uses mid price (bid + ask) / 2 instead of lastPrice to avoid stale trade prices.
        Interpolates total variance w = sigma^2 * T rather than implied vol directly,
        as total variance is the natural quantity for Dupire's local vol formula.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
        maturity_min (float, optional): Minimum time to maturity in years. Defaults to 0.1.
        maturity_max (float, optional): Maximum time to maturity in years. Defaults to 2.
        moneyness_min (float, optional): Minimum moneyness (strike/spot) to consider.
                                         Defaults to 0.85.
        moneyness_max (float, optional): Maximum moneyness (strike/spot) to consider.
                                         Defaults to 1.15.
        option_type (str, optional): The type of option ('call' or 'put'). Defaults to 'call'.
        max_spread_ratio (float, optional): Maximum allowed ask/bid ratio. Defaults to 3.0.
        min_volume (int, optional): Minimum option trading volume. Defaults to 10.

    Returns:
        tuple:
            - pd.DataFrame: A pivot table with TTM as index, strike prices as columns,
                          and total implied variance (sigma^2 * T) as values.
                          Missing values are NaN, not zero.
            - float: The current spot price of the underlying asset.

    Raises:
        ValueError: If option_type is not 'call' or 'put'.
        ValueError: If no valid options remain after filtering.
    """
    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    option_data = yf.Ticker(ticker)
    spot_price = option_data.history("1D")["Close"].iloc[0]
    today = pd.Timestamp.today()

    # Filter maturities within the specified range
    valid_maturities = [
        mat
        for mat in option_data.options
        if maturity_min < (pd.to_datetime(mat) - today).days / 365 < maturity_max
    ]

    all_data = []

    for maturity in valid_maturities:
        # Fetch the appropriate option chain
        if option_type == "call":
            chain = option_data.option_chain(maturity).calls
        else:
            chain = option_data.option_chain(maturity).puts

        ttm = (pd.to_datetime(maturity) - today).days / 365

        # Filter 1 — remove zero bids (illiquid options)
        chain = chain[chain["bid"] > 0]

        # Filter 2 — remove wide bid-ask spreads (unreliable pricing)
        chain = chain[(chain["ask"] / chain["bid"]) < max_spread_ratio]

        # Filter 3 — remove low volume options (stale prices)
        chain = chain[chain["volume"] >= min_volume]

        # Filter 4 — apply moneyness filter for numerical stability
        chain = chain[
            (chain["strike"] >= moneyness_min * spot_price)
            & (chain["strike"] <= moneyness_max * spot_price)
        ]

        # Use mid price instead of lastPrice to avoid stale trade prices
        chain["mid_price"] = (chain["bid"] + chain["ask"]) / 2

        # Filter 5 — remove Yahoo Finance IV outliers
        chain = chain[
            (chain["impliedVolatility"] > 0.05) & (chain["impliedVolatility"] < 2.0)
        ]

        # Compute total implied variance w = sigma^2 * T
        # This is the natural interpolation quantity for Dupire's formula
        chain["total_variance"] = chain["impliedVolatility"] ** 2 * ttm

        # Filter 6 — remove non-positive total variance
        chain = chain[chain["total_variance"] > 0]

        chain["TTM"] = ttm
        all_data.append(chain[["strike", "mid_price", "total_variance", "TTM"]])

    if not all_data:
        raise ValueError(
            "No valid options data after filtering. "
            "Consider relaxing moneyness or volume constraints."
        )

    combined = pd.concat(all_data, ignore_index=True)

    # Build pivot table with NaN for missing strike-maturity combinations
    # NaN is used instead of 0 to avoid corrupting interpolation downstream
    pivot_table = combined.pivot_table(
        index="TTM", columns="strike", values="mid_price", fill_value=np.nan
    )

    # Drop maturities with less than 50% valid strikes
    pivot_table = pivot_table.dropna(thresh=int(0.5 * pivot_table.shape[1]))

    # Drop strikes with less than 50% valid maturities
    pivot_table = pivot_table.dropna(axis=1, thresh=int(0.5 * pivot_table.shape[0]))

    return pivot_table, spot_price


if __name__ == "__main__":
    TICKER = "AAPL"
    table, spot = get_option_data(TICKER)
    print(f"Spot price: {spot:.2f}")
    print(f"Surface shape: {table.shape}")
    print(table)

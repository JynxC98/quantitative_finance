"""
Script to collect the relevant option data from yahoo finance.
"""

# import pandas as pd
import yfinance as yf


def get_option_data_from_yahoo(ticker, maturity_date, option_type="call"):
    """
    Fetches the option data from yahoo finance server.

    Input parameters
    ----------------
    ticker: The stock ticker.
    maturity_date: Option's maturity date (YYYY-MM-DD).
    option_type: Takes two values, `call` and `put`.

    Returns
    -------
        - option_data
            - dtype: pd.DataFrame
    """
    company_data = yf.Ticker(ticker)
    expiration_dates = company_data.options
    if maturity_date not in expiration_dates:
        raise ValueError(f"Please select the expiration date from {expiration_dates}")

    if option_type not in ("call", "put"):
        raise ValueError("Incorrect type of option value")

    option_data = company_data.option_chain(maturity_date)
    if option_type == "call":
        return option_data.calls

    return option_data.puts


if __name__ == "__main__":
    TICKER = "AAPL"
    data = get_option_data_from_yahoo(TICKER, "2024-04-12")
    print(data)

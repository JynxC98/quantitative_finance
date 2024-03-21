"""
A script to calculate local volatility function.
"""

import numpy as np


def calculate_local_volatility(
    initial_stock_price, max_stock_price, strike, initial_vol, time_to_mat, **params
):
    """
    Calculates local volatility based on Dupire's local volatility function.
    Uses finite difference approximation to calculate the local volatility.

    Input Parameters
    ----------------
    initial_stock_price: Price of the underlying at t = 0.
    strike: The strike price of the option.
    initial_vol: Volatility at t = 0.0.
    time_to_mat: Time to maturity, expressed in terms of years.

    Optional Parameters
    -------------------
    M: Number of spatial grid points.
    N: Number of time grid points.

    Output:
    -------
        - volatility grid
            dtype: np.ndarray
    """

    num_spatial_grid = params.get("M", 10000)  # Spatial grid points
    num_time_grid = params.get("N", 10000)  # Time grid

    dt = time_to_mat / num_time_grid  # Time step
    dS = max_stock_price / num_spatial_grid  # Spatial step

    stock_price_grid = np.linspace(
        initial_stock_price, max_stock_price, num_spatial_grid
    )
    volatility_grid = initial_vol * np.ones([num_spatial_grid, num_time_grid])

    for i in range(1, num_spatial_grid):
        for j in range(1, num_time_grid):
            return

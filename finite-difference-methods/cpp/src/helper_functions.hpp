/**
 * @file helper_functions.hpp
 *
 * @brief This file contains the helper functions used for the American option
 * pricing engine.
 *
 * @author Harsh Parikh
 * @date 11th October 2025
 */
#if !defined(HELPER_FUNCTIONS_HPP)
#define HELPER_FUNCTIONS_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

/**
 * @brief Calculate the cumulative distribution function (CDF) of a standard normal distribution
 * @param x: The input value for which the CDF is calculated.
 *
 * @returns The value of the standard normal CDF at x.
 */
template <typename T>
T normcdf(T x)
{
    return 0.5 * erfc(-x * M_SQRT1_2); // M_SQRT1_2 is 1/sqrt(2)
}

/**
 * @brief Calculate the option price using the Black-Scholes model. The main
 * functionality of this method is for benchmarking.
 *
 * @param spot: Current spot price.
 * @param strike: The predetermined strike price
 * @param T: The time to maturity.
 * @param r: The risk-free rate.
 * @param sigma: The current volatility.
 * @param isCall(optional): 1 if call option, else 0 for put option.
 *
 * @returns The option value.
 */
template <typename T>
T calculateOptionPrice(T spot, T strike, T ttm, T r, T q, T sigma, bool isCall)
{
    T d1 = (log(spot / strike) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(ttm));
    T d2 = d1 - sigma * sqrt(ttm);

    T option_value;

    // As per the option pricing theory, the value of the spot price decreases by
    // the dividend amount to ensure no arbitrage pricing takes place.

    option_value = isCall ? (spot * exp(-q * ttm) * normcdf(d1) - strike * exp(-r * T) * normcdf(d2))
                          : (strike * exp(-r * T) * normcdf(-d2) - spot * exp(-q * ttm) * normcdf(-d1));

    return option_value;
}

/**
 * @brief This function creates an equidistant grid between two ranges.
 * @param N: Number of grid points.
 * @param scaler_min: The starting point of the grid.
 * @param scaler_max: The end point of the grid.
 *
 * @returns A vector containing the equidistant grid points.
 */
template <typename T>
std::vector<T> getEquidistantGrid(int N, T scaler_min, T scaler_max)
{
    // Performing sanity checks.
    if (N < 2)
    {
        throw std::invalid_argument("There should be at least 2 values to create a grid");
    }

    if (scaler_min >= scaler_max)
    {
        throw std::invalid_argument("The value of scaler_min should be less than scaler_max");
    }
    // Initialising the grid with N data points.
    std::vector<T> result(N);

    // Calculating the range of the data points.
    T difference = scaler_max - scaler_min;

    // Calculating the grid value for the data points.
    T grid_value = difference / (N - 1);

    // Initialising the first value of the data grid.
    result[0] = scaler_min;

    for (int i = 1; i < N; ++i)
    {
        result[i] = result[i - 1] + grid_value;
    }
    return result;
}
#endif
/**
 * @brief This function acts as a helper to the `main.cpp` file that is used to
 * simulate the Merton Jump-Diffusion process.
 *
 * @author Harsh Parikh
 * @date 16th November 2025
 */

#ifndef HELPER_HPP
#define HELPER_HPP

#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Create an equidistant grid of points in [lower, upper].
 *
 * The returned vector contains `n_points` values starting at `lower` and ending
 * at `upper` (inclusive). Points are spaced uniformly. The function requires
 * n_points >= 2 and upper > lower.
 *
 * @param lower      Left endpoint of the interval.
 * @param upper      Right endpoint of the interval.
 * @param n_points   Number of grid points to produce (must be >= 2).
 * @return std::vector<double>  Vector of length n_points containing the grid.
 *
 * @throws std::invalid_argument if n_points < 2 or upper <= lower.
 */
std::vector<double> createEquidistantGrid(double lower, double upper, std::size_t n_points)
{
    if (n_points < 2)
    {
        throw std::invalid_argument("createEquidistantGrid: n_points must be >= 2");
    }
    if (!(upper > lower))
    {
        throw std::invalid_argument("createEquidistantGrid: upper must be greater than lower");
    }

    std::vector<double> grid;
    grid.reserve(n_points);

    const double step = (upper - lower) / static_cast<double>(n_points - 1);
    for (std::size_t i = 0; i < n_points; ++i)
    {
        grid.push_back(lower + static_cast<double>(i) * step);
    }

    // Ensure last point is exactly `upper` to avoid accumulated rounding error
    grid.back() = upper;

    return grid;
}

/**
 * @brief This function calculates the theoretical value of the Black-Scholes
 * option contract. The input parameters are as follows:
 * @param spot: The current spot price
 * @param strike: The pre-determined strike
 * @param r: Risk-free rate
 * @param sigma: The volatility of the underlying
 * @param T: Time to maturity
 * @param isCall: True for call and false for put
 */
double BlackScholesPrice(
    double spot,
    double strike,
    double r,
    double sigma,
    double T,
    bool isCall)
{

    // Generating the Normal CDF
    auto N = [](double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2));
    };

    // Calculating d1 and d2
    double d1 = (std::log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / sigma * std::sqrt(T);

    double d2 = d1 - std::sqrt(T) * sigma;

    // The sign is 1 for call option and -1 for put option. This is to ensure that
    // no if-else clause is used.

    int sign = isCall ? 1 : -1;

    double price = sign * spot * N(sign * d1) - sign * strike * std::exp(-r * T) * N(sign * -d2);

    return price;
}

#endif
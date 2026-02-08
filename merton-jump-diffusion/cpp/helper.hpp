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

#endif
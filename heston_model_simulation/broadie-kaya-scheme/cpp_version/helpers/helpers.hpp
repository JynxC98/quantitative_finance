/**
 * @brief This script stores the required functions for simulation.
 *
 * @author Harsh Parikh
 */

#if !defined(HELPERS_HPP)
#define HELPERS_HPP

#include <iostream>
#include <cmath>
#include <sstream>
#include <cassert>

#include <complex>
#include "gamma.hpp"

constexpr double EPS = 1e-8;

bool approx_equal(double a, double b, double tol = EPS)
{
    return std::abs(a - b) < tol;
}

// These parameters should be passed throughout the implementation of the
// characteristic function. For documentation, plese refer to the `bessel.hpp`
// file.

struct BesselFunctionParams
{
    std::complex<double> z;
    double alpha;
    int num_iterations;
    double tolerance;
    double threshold;
    bool log_space;
};

std::vector<double> getLinSpace(double lower_val, double upper_val, int N)
{
    // Initialising an empty array of gridpoints.
    std::vector<double> grid_points;

    // Base condition (Excluding negative elements for the sake of simplicity)
    if (N <= 1)
    {
        // If N <= 1, just return lower_val
        grid_points.push_back(lower_val);
        return grid_points;
    }

    // Calculating the step size
    double step = (upper_val - lower_val) / (N - 1);

    for (int i = 0; i < N; ++i)
    {
        grid_points.push_back(lower_val + i * step);
    }

    return grid_points;
}

/**
 * @brief This function calculates the next greatest power of 2 using the
 * bitwise left shift operator.
 */
int next_power_of_two(int n)
{
    int res = 1;
    while (res < n)
        res <<= 1; // This code shifts the bit to the left until the
                   // power of two is greater than the input number.
    return res;
}

/**
 * @brief This function is used to fetch linearly seperated numbers between the
 * two sets of inputs.
 *
 * @param a: The lower val
 * @param b: The upper val
 * @param num_points: The number of points required.
 */
template <typename T>
std::vector<T> getLinspace(T a, T b, int num_points)
{
    assert(num_points > 0 && "The number of data points must be positive");

    std::vector<T> result;
    result.reserve(num_points);

    if (num_points == 1)
    {
        result.push_back(a);
        return result;
    }

    T step = (b - a) / static_cast<T>(num_points - 1);

    for (int i = 0; i < num_points; ++i)
    {
        result.push_back(a + i * step);
    }

    return result;
}

double linear_interpolate(double K,
                          const std::vector<double> &x,
                          const std::vector<double> &y)
{
    // Bounds check
    if (K <= x.front())
        return y.front();
    if (K >= x.back())
        return y.back();

    // Binary search for index such that: x[i] <= K < x[i+1]
    auto upper = std::upper_bound(x.begin(), x.end(), K);
    size_t idx = std::distance(x.begin(), upper) - 1;

    double x0 = x[idx];
    double x1 = x[idx + 1];
    double y0 = y[idx];
    double y1 = y[idx + 1];

    // Linear interpolation formula
    double weight = (K - x0) / (x1 - x0);
    return y0 + weight * (y1 - y0);
}

#define ASSERT(condition, message)                                                \
    do                                                                            \
    {                                                                             \
        if (!(condition))                                                         \
        {                                                                         \
            std::cerr << "Assertion failed: " << message                          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            std::terminate();                                                     \
        }                                                                         \
    } while (false)

#endif
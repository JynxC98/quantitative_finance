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
#include <vector>
#include <cmath>

#include <complex>
#include "gamma.hpp"

constexpr double EPS = 1e-8;

inline bool approx_equal(double a, double b, double tol = EPS)
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

/**
 * @brief This function calculates the next greatest power of 2 using the
 * bitwise left shift operator.
 */
inline int next_power_of_two(int n)
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
inline std::vector<T> getLinspace(T a, T b, int num_points)
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

/**
 * @brief This function is used to linearly interpolate the value between two
 * datapoints.
 *
 * @param K: The value at which the interpolation is required.
 * @param x: The x-axis grid
 * @param y: The y-axis grid
 *
 * @returns: The linearly interpolated value
 */
inline double linear_interpolate(double K,
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

struct StatisticalProperties
{
    double mean;
    double std_dev;
    double left_lc;
    double right_lc;
};

/**
 * @brief This function is used to calculate the mean, standard deviation, left
 * confidence interval and right confidence interval of a underlying vector using
 * the Welford's algorithm. The function inherently assumes 95% confidence interval.
 *
 * The input parameters are as follows:
 *
 * @param vals: The vector of the underlying input values
 */
inline StatisticalProperties calculateStatistics(const std::vector<double> &vals)
{
    StatisticalProperties results;

    int n = 0;
    double mean = 0.0;
    double M2 = 0.0;

    // Welford's online algorithm
    for (const double x : vals)
    {
        n++;
        double delta = x - mean; // The difference before the mean updates
        mean += delta / n;

        // `delta2` is the difference after multiplying them accumulates the variance numerator stably,
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    // n - 1 for sample variance
    // Uses Bessel's correction since your Monte Carlo paths are a sample, not the full population.

    double std_dev = std::sqrt(M2 / (n - 1));

    // 95% CI: mean +/- 1.96 * (std_dev / sqrt(n))
    double margin = 1.96 * (std_dev / std::sqrt(static_cast<double>(n)));

    results.mean = mean;
    results.std_dev = std_dev;
    results.left_lc = mean - margin;
    results.right_lc = mean + margin;

    return results;
}

inline double BlackScholesPrice(double spot,
                                double strike,
                                double sigma,
                                double r,
                                double T,
                                bool isCall)
{
    auto N = [](double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    double d1 = (std::log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

    double d2 = d1 - sigma * std::sqrt(T);

    if (isCall)
        return spot * N(d1) - strike * std::exp(-r * T) * N(d2);
    else
        return strike * std::exp(-r * T) * N(-d2) - spot * N(-d1);
}
#endif
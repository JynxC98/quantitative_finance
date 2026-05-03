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
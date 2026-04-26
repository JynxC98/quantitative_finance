/**
 * @brief This script stores the required functions for simulation.
 *
 * @author Harsh Parikh
 */

#if !defined(HELPERS_HPP)
#define HELPERS_HPP

#include <iostream>
#include <cmath>
#include <complex>
#include "gamma.hpp"

constexpr double EPS = 1e-8;

bool approx_equal(double a, double b, double tol = EPS)
{
    return std::abs(a - b) < tol;
}

#endif
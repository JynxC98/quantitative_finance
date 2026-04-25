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

/**
 * @brief This function stores calculates the value of the modified Bessel's
 * function for the value of `z`. The function uses power series for the small
 * value of |z| and asymptomatic expansion for the large value of |z| depending
 * on the threshold.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param upper_lim: The upper value for the summation.
 */
std::complex<double> ModifiedBessel(std::complex<double> z,
                                    double alpha,
                                    int upper_lim = 1000,
                                    double tolerance = 1e-12,
                                    double threshold = 10.0) // Removed semicolon here
{
    // Base cases for zero values
    // For real order α:
    // I_0(0) = 1 (by analytic continuation)
    // I_α(0) = 0 for α > 0
    // I_α(0) diverges (→ ∞) for α < 0

    auto abs_value = [](std::complex<double> num)
    {
        return std::sqrt(std::real(num) * std::real(num) +
                         std::imag(num) * std::imag(num));
    };

    if (z == std::complex<double>(0.0, 0.0))
    {

        if (alpha < 0.0)
        {
            throw std::domain_error("I_alpha(0) diverges to infinity for alpha < 0");
        }
        if (alpha == 0.0)
        {
            return std::complex<double>(1.0, 0.0); // I_0(0) = 1
        }
        // alpha > 0
        return std::complex<double>(0.0, 0.0);
    }

    std::complex<double> result(0.0, 0.0);
    std::complex<double> term(1.0, 0.0);

    // First term: k=0
    term = 1.0 / GammaFunction(alpha + 1.0);
    result = term;

    // Subsequent terms using recurrence
    for (int k = 1; k < upper_lim; ++k)
    {
        // term_k = term_{k-1} * (z^2/4) / (k * (alpha + k))
        term *= (0.25 * z * z) / (static_cast<double>(k) * (alpha + static_cast<double>(k)));
        result += term;

        // Check convergence
        if (std::abs(term) < tolerance * std::abs(result))
        {
            break;
        }
    }

    // Multiply by (z/2)^α
    return std::pow(0.5 * z, alpha) * result;
}

#endif
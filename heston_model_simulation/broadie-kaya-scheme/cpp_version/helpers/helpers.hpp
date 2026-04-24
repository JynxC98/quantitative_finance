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
 * function for the value of `z``
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param upper_lim: The upper value for the summation.
 */
std::complex<double> ModifiedBessel(std::complex<double> z,
                                    std::complex<double> alpha,
                                    int upper_lim = 1000)
{
    // Basecase for the Bessel's function
    if (z == std::complex<double>(0.0, 0.0))
    {
        // I_alpha(0) = 1 if alpha = 0 else 0
        if (alpha == std::complex<double>(0.0, 0.0))
        {
            return std::complex<double>(1.0, 0.0);
        }
        return std::complex<double>(0.0, 0.0);
    }

    std::complex<double> result(0.0, 0.0);

    for (int i = 0; i < upper_lim; ++i)
    {

        // std:: tgamma(i+1) is used as i! for integer i

        result += pow((0.25 * z * z), i) / (std::tgamma(i + 1) * GammaFunction(alpha +
                                                                               static_cast<double>(i) + 1.0));
    }

    // Uses principle branch of the complex logarithm
    return std::exp(alpha * std::log(0.5 * z)) * result;
}

#endif
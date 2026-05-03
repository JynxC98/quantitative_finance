/**
 * @brief This script is used to approximate the value of Gamma distribution
 * at a particular value. More details on the approximation can be found here:
 *
 * https://en.wikipedia.org/wiki/Lanczos_approximation
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <cmath>
#include <complex>

#include "../helpers/gamma.hpp"

std::complex<double> GammaFunction(std::complex<double> z)
{

    // Reflection formula for Re(z)
    if (std::real(z) < 0.5)
    {
        return M_PI / (std::sin(M_PI * z) * GammaFunction(1.0 - z));
    }

    // Lanczos approximation
    z -= 1.0;
    std::complex<double> x = p[0];

    for (size_t i = 1; i < p.size(); ++i)
    {
        x += p[i] / (z + static_cast<double>(i));
    }
    std::complex<double> t = z + static_cast<double>(g) + 0.5;

    return std::sqrt(2.0 * M_PI) *
           std::pow(t, z + 0.5) *
           std::exp(-t) * x;
}

// Operator overloading
double GammaFunction(double z)
{
    auto result = GammaFunction(std::complex<double>(z, 0.0));
    return std::real(result);
}

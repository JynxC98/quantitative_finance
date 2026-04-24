/**
 * @brief This script is used to approximate the value of Gamma distribution
 * at a particular value. More details on the approximation can be found here:
 *
 * https://en.wikipedia.org/wiki/Lanczos_approximation
 *
 * @author Harsh Parikh
 */

#if !defined(GAMMA_HPP)
#define GAMMA_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>

// Adding the constant parameter values of the Lanczos' approximation.

const static int g = 8;
const static int n = 12;
const static std::vector<double> p = {
    0.9999999999999999298,
    1975.3739023578852322,
    -4397.3823927922428918,
    3462.6328459862717019,
    -1156.9851431631167820,
    154.53815050252775060,
    -6.2536716123689161798,
    0.034642762454736807441,
    -7.4776171974442977377e-7,
    6.3041253821852264261e-8,
    -2.7405717035683877489e-8,
    4.0486948817567609101e-9};

/**
 * @brief Gamma function approximation using Lanczos.
 *
 * @param z: Input value (real or complex)
 * @return Approximate Gamma(z)
 */
template <typename T>
std::complex<double> GammaFunction(std::complex<T> z)
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

/**
 * @brief Overload for real inputs
 */
inline double GammaFunction(double z)
{
    auto result = GammaFunction(std::complex<double>(z, 0.0));
    return std::real(result);
}
#endif
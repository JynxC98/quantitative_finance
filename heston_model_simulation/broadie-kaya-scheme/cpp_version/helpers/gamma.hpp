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

std::complex<double> GammaFunction(std::complex<double> z);

/**
 * @brief Overload for real inputs
 */
double GammaFunction(double z);
#endif
/**
 * @brief This script stores the characteristic function of the conditional
 * variance for the Heston Model simulation.
 *
 * @author Harsh Parikh
 */
#if !defined(CHAR_FUNCTION_HPP)
#define CHAR_FUNCTION_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

#include "bessel.hpp"

/**
 * @brief This struct format acts as a container for Heston parameters
 *
 * @param kappa: The mean reversion rate
 * @param theta: The long run average
 * @param sigma: Volatility of variance
 * @param v_u: The variance value at timestep u
 * @param v_t: The variance value at timestep t (t > u)
 * @param dt: The value of timestep, t-u
 */
struct HestonParams
{
    double kappa;
    double theta;
    double sigma;
    double v_u;
    double v_t;
    double dt;
};

/**
 * @brief The characteristic function of the conditional variance. The formulation
 * of the characteristic function is given as follows:
 *
 * \Phi(a) = E[exp(ia \int_u^t v_s ds) | v_u, v_t]
 *
 * The goal is to find the area under the curve between the points u and t (t > u)
 * for the stochastic evolution of v_t
 *
 * @param p: The Heston parameters defined as the structure above.
 * @param u: The characteristic variable
 */
std::complex<double> CharFunction(const HestonParams &p, double u);

#endif
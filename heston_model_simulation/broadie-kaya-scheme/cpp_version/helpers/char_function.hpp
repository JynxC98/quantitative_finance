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
std::complex<double> CharFunction(const HestonParams &p, double u)
{
    // Initialising the complex number
    std::complex<double> i(0.0, 1.0);

    // Guard for the u=0 singularity to prevent 0/0 (NaN)
    if (std::abs(u) < 1e-12)
    {
        return std::complex<double>(1.0, 0.0);
    }

    /*
    The characteristic function comprises of 3 terms.
    */

    // The const gamma function is as follows:
    // const_gamma(u) = sqrt(kappa^2 - 2 * sigma^2 * i * u)
    std::complex<double> const_gamma = std::sqrt(p.kappa * p.kappa -
                                                 2.0 * p.sigma * p.sigma * i * u);

    // Precomputing common exponentials for efficiency and clarity
    std::complex<double> exp_kappa_dt = std::exp(-p.kappa * p.dt);
    std::complex<double> exp_gamma_dt = std::exp(-const_gamma * p.dt);

    // Evaluating the first term
    // first_term = (gamma * exp(-0.5*(gamma-kappa)*dt) * (1-exp(-kappa*dt))) / (kappa * (1-exp(-gamma*dt)))
    std::complex<double> first_term = (const_gamma * std::exp(-0.5 * (const_gamma - p.kappa) * p.dt) *
                                       (1.0 - exp_kappa_dt)) /
                                      (p.kappa * (1.0 - exp_gamma_dt));

    // Evaluating the second term
    std::complex<double> second_term = std::exp(((p.v_u + p.v_t) / (p.sigma * p.sigma)) *
                                                ((p.kappa * (1.0 + exp_kappa_dt)) / (1.0 - exp_kappa_dt) -
                                                 (const_gamma * (1.0 + exp_gamma_dt)) / (1.0 - exp_gamma_dt)));

    // Evaluating the third term
    double d = 4.0 * p.kappa * p.theta / (p.sigma * p.sigma);
    double alpha = 0.5 * d - 1.0;

    std::complex<double> bessel_arg_num = std::sqrt(p.v_u * p.v_t) *
                                          ((4.0 * const_gamma * std::exp(-0.5 * const_gamma * p.dt)) /
                                           (p.sigma * p.sigma * (1.0 - exp_gamma_dt)));

    std::complex<double> bessel_arg_den = std::sqrt(p.v_u * p.v_t) *
                                          ((4.0 * p.kappa * std::exp(-0.5 * p.kappa * p.dt)) /
                                           (p.sigma * p.sigma * (1.0 - exp_kappa_dt)));

    /*
     * Evaluating the ratio of Modified Bessel functions.
     */
    std::complex<double> numerator = ModifiedBessel(bessel_arg_num, alpha);
    std::complex<double> denominator = ModifiedBessel(bessel_arg_den, alpha);

    std::complex<double> third_term = numerator / denominator;

    return first_term * second_term * third_term;
}

#endif
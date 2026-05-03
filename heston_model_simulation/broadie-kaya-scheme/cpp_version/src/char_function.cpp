/**
 * @brief This script stores the characteristic function of the conditional
 * variance for the Heston Model simulation.
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <cmath>
#include <complex>

#include "../helpers/gamma.hpp"
#include "../helpers/bessel.hpp"
#include "../helpers/char_function.hpp"

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
                                                (((p.kappa * (1.0 + exp_kappa_dt)) / (1.0 - exp_kappa_dt)) -
                                                 ((const_gamma * (1.0 + exp_gamma_dt)) / (1.0 - exp_gamma_dt))));

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

    // Hard coding the parameters for Modified Bessel. Future fix will include
    // Incorporating the variables as input parameters

    int num_iterations = 100;
    double tolerance = 1e-8;
    double threshold = 10.0;
    bool log_space = true;
    std::complex<double> log_numerator = ModifiedBessel(bessel_arg_num, alpha,
                                                        num_iterations, tolerance,
                                                        threshold, log_space);

    std::complex<double> log_denominator = ModifiedBessel(bessel_arg_den, alpha,
                                                          num_iterations, tolerance,
                                                          threshold, log_space);

    std::complex<double> third_term = std::exp(log_numerator - log_denominator);

    return first_term * second_term * third_term;
}

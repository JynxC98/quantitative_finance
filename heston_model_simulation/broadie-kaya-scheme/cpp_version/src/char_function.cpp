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
#include "../helpers/bessel_parameters.hpp"
#include "../helpers/heston_params.hpp"

std::complex<double> CharFunction(const HestonParams &p, std::complex<double> u)
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

    BesselParams params; // These params are used for bessel params.

    int num_iterations = params.num_iterations;
    double tolerance = params.tolerance;
    double threshold = params.threshold;
    bool log_space = params.log_space;

    std::complex<double> log_numerator = ModifiedBessel(bessel_arg_num, alpha,
                                                        num_iterations, tolerance,
                                                        threshold, log_space);

    std::complex<double> log_denominator = ModifiedBessel(bessel_arg_den, alpha,
                                                          num_iterations, tolerance,
                                                          threshold, log_space);

    // Branch-cut correction: as u sweeps over the integration range used by
    // the Gil-Pelaez inversion, bessel_arg_num spirals around the origin
    // (its phase is dominated by the unbounded term -0.5*Im(const_gamma)*dt
    // from exp(-0.5*const_gamma*dt), while |1 - exp_gamma_dt| stays pinned
    // near 1 with negligible phase). ModifiedBessel's log-space leading
    // factor (z/2)^alpha is computed via std::arg, which always wraps to
    // (-pi, pi], so every time the true phase crosses a branch cut the
    // computed log jumps by a spurious +-2*pi*i -- and since alpha is
    // non-integer here (alpha = 2*kappa*theta/sigma^2 - 1), that jump does
    // NOT cancel in exp(alpha * i * 2*pi), corrupting phi(u) at each
    // crossing. This showed up as a non-monotonic reconstructed CDF /
    // negative reconstructed PDF, confirmed unrelated to quadrature
    // resolution or truncation frequency (both already converged). Recover
    // the true continuous phase analytically and correct the discrepancy.
    double wrapped_arg = std::arg(bessel_arg_num);
    double continuous_arg = std::arg(const_gamma) - 0.5 * const_gamma.imag() * p.dt -
                            std::arg(1.0 - exp_gamma_dt);
    log_numerator += std::complex<double>(0.0, alpha * (continuous_arg - wrapped_arg));

    std::complex<double> third_term = std::exp(log_numerator - log_denominator);

    return first_term * second_term * third_term;
}

std::complex<double> CharFunction(const HestonParams &p, double u)
{
    return CharFunction(p, std::complex<double>(u, 0.0));
}

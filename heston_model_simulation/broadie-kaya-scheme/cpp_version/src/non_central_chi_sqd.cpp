/**
 * @brief This function is used to evaluate non-central chi squared distribution
 * for a particular value. More details on the same can be accessed here:
 *
 * https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
 *
 * @author: Harsh Parikh
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>

#include "../helpers/bessel.hpp"

double NonCentralChi2PDF(double z,
                         double dof,
                         double lambda_)
{
    // Edge cases
    if (z <= 0.0)
        return 0.0; // PDF is zero for values leq 0.

    // Edge cases for negative values of lambda and dof
    if (dof <= 0.0)
        throw std::domain_error("Degrees of freedom must be positive");
    if (lambda_ < 0.0)
        throw std::domain_error("Non-centrality parameter must be non-negative");

    // Evaluating the exponential term
    double exp_term = 0.5L * std::exp(-0.5L * (z + lambda_));

    // Evaluating the centrality term (now safe since z > 0, lambda_ > 0)
    double cent_term = std::exp((dof / 4.0 - 0.5) * std::log(z / lambda_));

    // Evaluating the Bessel term
    double sqrt_term = std::sqrt(lambda_ * z);
    double alpha = 0.5 * dof - 1.0;

    // ModifiedBessel returns complex, but real part is the actual value

    /*
    Currently hardcoding the parameters, the actual parameters will be passed
    via input to the function
     */
    int num_iterations = 100;
    double tolerance = 1e-8;
    double threshold = std::abs(alpha) + 10.0;
    bool log_space = true;

    auto bessel_result = ModifiedBessel(std::complex<double>(sqrt_term, 0.0), alpha, num_iterations, tolerance,
                                        threshold, log_space);

    // Use the real part - imaginary should be ~0
    double bessel_term = std::real(std::exp(bessel_result));

    if (std::abs(std::imag(bessel_result)) > 1e-10)
    {
        std::cerr << "Warning: Non-zero imaginary part in NonCentralChi2PDF: "
                  << std::imag(bessel_result) << std::endl;
    }

    return exp_term * cent_term * bessel_term;
}

double SampleNonCentralChi2(double dof,
                            double lambda_)
{
    // Initialising the random number engine
    static std::random_device dev;
    static std::mt19937 rng(dev());

    // Edge cases for negative values of lambda and dof
    if (dof <= 0.0)
        throw std::domain_error("Degrees of freedom must be positive");
    if (lambda_ < 0.0)
        throw std::domain_error("Non-centrality parameter must be non-negative");

    /*
    The flow summary of generating a random Non-Central chi squared val is as
    follows:
    1. Calculate lambda based on the current state v_t (Used as the input value
    in the current implementation)
    2. Draw a random integer `n` from Poisson(lambda_/2.0).
    3. Draw a random float `x` from central chi^2(dof + 2n)

    This mixture approach avoids the numerical instability of evaluating the
    Bessel functions present in the direct PDF.
    */

    // Poisson Generator
    double lambda_new = lambda_ / 2.0;
    std::poisson_distribution<int> Poisson(lambda_new);
    int n = Poisson(rng);

    // A Central Chi-Squared(k) is equivalent to Gamma(shape=k/2, scale=2)
    double k_shape = (dof + 2.0 * n) / 2.0;
    double theta_scale = 2.0; // Scaling factor, usually kept as `2`in the implementation.

    // More details and references can be found in `CALCULATIONS.md`.

    // Step 3: Draw a sample from the Gamma distribution
    std::gamma_distribution<double> GammaDist(k_shape, theta_scale);

    /*
    We return the realization from the distribution. This value represents
    the stochastic component of the variance transition before the final
    Heston-specific scaling factor is applied.
    */
    return GammaDist(rng);
}

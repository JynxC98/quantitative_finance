/**
 * @brief This script is used to calculate the integrated variance for the
 * CIR process.
 *
 * @author Harsh Parikh
 */
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

#include "../helpers/heston_params.hpp"
#include "../helpers/non_central_chi_sqd.hpp"
#include "../helpers/solvers.hpp"
#include "../helpers/integrated_variance.hpp"
#include "../helpers/random_utils.hpp"

double calculateIntegratedVariance(const HestonParams &p)
{

    double U = uniform(gen);

    // Calculating the integrated variance
    double integrated_variance = runNewtonSolver(U, p);

    return integrated_variance;
}

double sampleVt(const HestonParams &p, double v_u)
{

    // Calculating the exponential factor
    double kappa_exp = std::exp(-p.kappa * p.dt);

    // Calculating the scaling factor
    double scaling_factor = (p.sigma * p.sigma * (1.0 - kappa_exp)) / (4.0 * p.kappa);

    // Calculating the dof for Non-Central CQ distribution
    double dof = 4.0 * p.kappa * p.theta / (p.sigma * p.sigma);

    // Calculating the non-centrality parameter
    double lambda_ = ((4.0 * p.kappa * kappa_exp) / (p.sigma * p.sigma * (1.0 - kappa_exp))) * v_u;

    return scaling_factor * SampleNonCentralChi2(dof, lambda_);
}

double VarianceBrownianIntegral(const HestonParams &p, double integrated_variance)
{

    double val = (1.0 / p.sigma) * (p.v_t - p.v_u - p.kappa * p.theta * p.dt + p.kappa * integrated_variance);
    return val;
}

double priceStep(const HestonParams &p, double S_u, double integrated_variance, double r)
{
    // Calculating the mean of the process

    // Calculating the log stock price
    double log_stock = std::log(S_u);

    // Calculating the variance Brownian integral
    double variance_brownian_integral = VarianceBrownianIntegral(p, integrated_variance);

    // Calculating the mean of the evolution process
    double mean = log_stock + r * p.dt - 0.5 * integrated_variance + p.rho * variance_brownian_integral;

    // Calculated the std-dev of the process
    double std_dev = (integrated_variance > 1e-12) ? std::sqrt((1.0 - p.rho * p.rho) * integrated_variance) : 0.0;

    // Fetching the standard normal variable
    double Z = normal(gen);

    double price = std::exp(mean + std_dev * Z);

    return price;
}
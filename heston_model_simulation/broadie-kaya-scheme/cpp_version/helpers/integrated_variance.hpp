/**
 * @brief This script serves as a header file for the function used to calculate
 * the integrated variance of the CIR process.
 *
 * @author Harsh Parikh
 */

#if !defined(INTEGRATED_VARIANCE_HPP)
#define INTEGRATED_VARIANCE_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <unordered_map>

#include "heston_params.hpp"
#include "solvers.hpp"

/**
 * @brief This function is used to calculate the integrated variance for the
 * CIR process.
 */
double calculateIntegratedVariance(const HestonParams &p);

/**
 * @brief Exact simulation of CIR variance process for Heston model.
 *
 * Samples v_t from its non-central chi-squared conditional distribution given v_u.
 *
 * @param p    Heston parameters (κ, θ, σ, Δt)
 * @param v_u  Current variance
 * @return     Next variance value (exact sample, no discretization error)
 */
double sampleVt(const HestonParams &p, double v_u);

/**
 * @brief Computes the variance Brownian integral analytically
 * using the CIR SDE identity.
 *
 * From integrating dV_t = kappa*(theta - V_t)dt + sigma*sqrt(V_t)*dW_t^V:
 *
 * int_u^t sqrt(V_s) dW_s^V = (1/sigma) * (V_t - V_u - kappa*theta*dt + kappa * int_var)
 *
 * @param p Heston parameters
 * @param integrated_variance The integrated variance int_u^t V_s ds
 * @return Value of the stochastic integral
 */
double VarianceBrownianIntegral(const HestonParams &p, double integrated_variance);

/**
 * @brief Computes the next log-price step in the Heston model using Euler discretization.
 *
 * This function advances the log-asset price from time u to time t given the
 * variance path and the integrated variance. It accounts for the correlation
 * between asset and variance processes through a correlated term derived from
 * the variance increment.
 *
 * @param p        Heston model parameters (r, rho, sigma, kappa, theta, dt)
 * @param log_s_u  Log-price at time u (current)
 * @param r:        The risk-free rate
 * @param int_var  Integrated variance from u to t
 * @param Z        Standard normal random variable
 *
 * @return         Asset price at time t (next step)
 */
double priceStep(const HestonParams &p, double log_s_u, double r,
                 double int_var, double Z);
#endif
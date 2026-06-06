/**
 * @brief This header file stores the implementation of the Euler's and Milstein's
 * scheme for simulating the Heston model's path.
 *
 * @author Harsh Parikh
 */

#if !defined(MODELS_HPP)
#define MODELS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include "heston_params.hpp"
#include "helpers.hpp"

/**
 * @brief This structure stores the information of the underlying option contract.
 * The parameter description are as follows:
 *
 * @param spot: The underlying spot price
 * @param strike: The predetermined strike price
 * @param r: The risk free rate
 * @param T: Time to maturityß
 */
struct OptionParams
{
    double spot;
    double strike;
    double r;
    double T;
};

// This class is used to select the overall volatility structure for the
// simulation process.
enum class VariancePrevention
{
    Reflection,
    Truncation
};

/**
 * @brief This structure stores the simulation results for the Heston model
 * simulation. The parameters are as follows:
 *
 * @param mean: The mean of the overall process
 * @param std_dev: The standard deviation of the process
 * @param left_lc: The left confidence interval
 * @param right_lc: The right confidence interval
 */
struct SimulationResults
{
    double mean;
    double std_dev;
    double left_lc;
    double right_lc;
};

/**
 * @brief This function is used to simulate the Euler's SDE scheme using the
 * Monte-Carlo simulation. The function allows two modifications for the negative
 * variance prevention: reflection and trucation,
 *
 * Reflection principle is given as follows:
 * variance_t = std::abs(variance_u)
 *
 * Truncation principle is given as follows:
 * variance_t = std::max(variance_u, 0.0)
 *
 * The input parameters are as follows:
 * @param p: The structure file of the Heston Model Parameters. Please refer to its
 * documentation for more details.
 * @param o: The structural file of the option price details. Please refer to its
 * documentation for more details.
 * @param M: The number of Monte-Carlo paths.
 * @param N: The number of time steps for the Monte-Carlo path
 * @param prevention: The negative variance prevention mechanism for the overall process.
 *
 * @returns: A map with left CL, right CL, mean, and std deviation of the process.
 */
StatisticalProperties EulerScheme(const HestonParams &p,
                                  const OptionParams &o,
                                  int M,
                                  int N,
                                  bool isCall,
                                  VariancePrevention prevention);

/**
 * @brief Simulates the Heston stochastic volatility model using the exact Broadie-Kaya scheme.
 *
 * This function implements the exact simulation scheme proposed by Broadie & Kaya (2006),
 * which simulates the Heston model without discretization bias. The algorithm uses the
 * non-central chi-squared distribution for variance transitions and the conditional
 * distribution for the integrated variance and asset price.
 *
 * @param p    Heston model parameters
 * @param o    Option contract parameters
 * @param M    Number of simulation paths
 * @param N    Number of time steps per path (discretization points)
 */
StatisticalProperties simulateBroadieKayaHeston(const HestonParams &p,
                                                const OptionParams &o,
                                                int M, int N,
                                                bool isCall,
                                                const std::string &cache_path = "cdf_cache.bin");

#endif
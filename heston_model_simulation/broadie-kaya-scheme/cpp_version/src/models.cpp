/**
 * @brief This script stores the overall implementation of the Euler's
 * and Milstein's scheme for simulating the underlying asset path.
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "../helpers/helpers.hpp"
#include "../helpers/heston_params.hpp"
#include "../helpers/models.hpp"

StatisticalProperties EulerScheme(const HestonParams &p,
                                  const OptionParams &o,
                                  int M,
                                  int N,
                                  bool isCall = true,
                                  VariancePrevention prevention = VariancePrevention::Truncation)
{
    // Fetching the seed value from the hardware for generating random numbers.
    std::random_device seed;

    // Initiazing the generator for the random numbers
    std::mt19937 gen(seed());

    // Initializng the standard normal distribution generator
    std::normal_distribution<> norm(0.0, 1.0);

    // We only need to store the results at the end path
    std::vector<double> option_prices(M, 0.0);

    // Calculating the value of the timestep

    double dt = o.T / static_cast<double>(N);

    // This variable stores the transformed variance based on the prevention
    // criteria
    double trans_var;

    // These variables store the standard normal variables
    double dW1, dW2;

    for (int i = 0; i < M; ++i)
    {
        // These variables store the evolution of the current spot and variance
        double current_spot = o.spot;   // The spot at t = 0
        double current_variance = p.v0; // The variance at t = 0

        for (int j = 1; j <= N; ++j)
        {
            // Initialzing the random variabbles dW1 and dW2 based on the
            // literature

            dW1 = norm(gen);

            dW2 = p.rho * dW1 + std::sqrt(1.0 - p.rho * p.rho) * norm(gen);

            switch (prevention)
            {

            case VariancePrevention::Reflection:

                trans_var = std::abs(current_variance);
                break;

            case VariancePrevention::Truncation:

                trans_var = std::max(current_variance, 0.0);
                break;
            }

            current_spot = current_spot * (1.0 + o.r * dt + std::sqrt(trans_var) * std::sqrt(dt) * dW1);
            current_variance = current_variance + p.kappa * (p.theta - current_variance) * dt + p.sigma * trans_var * std::sqrt(dt) * dW2;
        }

        // Storing the evolution of the path
        option_prices[i] = isCall ? std::max(current_spot - o.strike, 0.0)
                                  : std::max(o.strike - current_spot, 0.0);
    }

    auto results = calculateStatistics(option_prices);

    return results;
}
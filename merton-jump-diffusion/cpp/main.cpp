/**
 * @file main.cpp
 * @brief The following script is an implementation of the Merton's
 * Jump-Diffusion ("MJD") framework. MJD model is extends the traditional
 * Black-Scholes model to capture the negative skewness and excess kurtosis
 * of the log stock price densitiy by addition of the compound Poisson jump
 * process.
 *
 * @author Harsh Parikh
 * @date 8th February 2026
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>

#include "helper.hpp"

/**
 * @brief This structure acts as a skeleton for the framework of the overall Jump-
 * Diffusion model. The parameters are as follows:
 * @param spot: The current spot price
 * @param strike: The pre-determined strike price
 * @param T: Time to maturity
 * @param r: The risk free rate
 * @param sigma: The volatility of the underlying
 * @param muJ: The mean of the Jump Diffusion framework
 * @param sigmaJ: The standard deviation of the Jump Diffusion framework
 * @param lambdaJ: The intensity of the Jump
 */
struct MertonJumpDiffusion
{
    double spot;
    double strike;
    double T;
    double r;
    double sigma;
    double muJ;
    double sigmaJ;
    double lambdaJ;
};

/**
 * @brief The main function to calculate the price of an option using the
 * Merton's JD framework. The price of the call option is calculated under the risk-
 * neutral framework. The functions takes in the following parameters:
 * @param p: The struct input of the MJD parameters
 * @param M: The number of Monte-Carlo paths
 * @param N: The number of time steps
 * @param isCall: true for call and false for put
 * @returns The value of the underlying option.
 */

std::map<std::string, double> CalculateOptionPrice(const MertonJumpDiffusion &p, int M, int N, bool isCall)
{
    // Initializing the random number generator.
    std::random_device dev;
    std::mt19937 rng(dev());

    // Calculating the time step
    double dt = p.T / static_cast<double>(N);

    // Calculating the value of `k`term
    double k = std::exp(p.muJ + 0.5 * p.sigmaJ * p.sigmaJ) - 1;

    // Developing the overall grid
    std::vector<double> grid(M, 0.0);

    // Generating Normal distribution
    std::normal_distribution norm(0.0, 1.0);

    // Log-normal jump distribution
    std::normal_distribution dJ(p.muJ, p.sigmaJ);

    // Generating Poission's distribution
    std::poisson_distribution poisson(p.lambdaJ * dt);
    // Populating the grid
    for (int m = 0; m < M; ++m)
    {
        // Calculating the drift of the path.
        // The drift will remain the same throughout the paths.
        auto drift = (p.r - 0.5 * (p.sigma * p.sigma) - p.lambdaJ * k) * dt;

        double running_spot_log = std::log(p.spot);

        for (int n = 1; n <= N; ++n)
        {
            // Calculating the random variables
            // Generating the Brownian increment.

            double dZ = norm(rng);

            // Calculating the number of Jumps
            int num_jumps = poisson(rng);

            // Calculating the `Y_t`term
            double Y_t = 0.0;

            for (int i = 0; i < num_jumps; ++i)
            {

                double jump = dJ(rng);
                Y_t += jump;
            }

            running_spot_log = running_spot_log + drift + p.sigma * std::sqrt(dt) * dZ + Y_t;
        }

        grid[m] = running_spot_log;
    }

    // Fetching the payoff

    double mean = 0.0;
    double M2 = 0.0; // sum of squared deviations

    for (int m = 0; m < M; ++m)
    {
        double final_price = std::exp(grid[m]);

        double payoff = isCall
                            ? std::max(final_price - p.strike, 0.0)
                            : std::max(p.strike - final_price, 0.0);

        double delta = payoff - mean;
        mean += delta / (m + 1);
        M2 += delta * (payoff - mean);
    }

    double variance = M2 / (M - 1);
    double std_dev = std::sqrt(variance);
    double std_error = std_dev / std::sqrt(M);

    double discount = std::exp(-p.r * p.T);
    double discounted_mean = mean * discount;
    double discounted_std_error = std_error * discount;

    double left_error = discounted_mean - 1.96 * discounted_std_error;
    double right_error = discounted_mean + 1.96 * discounted_std_error;

    std::map<std::string, double> results;

    results["Mean Price"] = discounted_mean;
    results["Std Dev"] = std_dev;
    results["Left error"] = left_error;
    results["Right error"] = right_error;

    return results;
}

int main()
{
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.045;
    double sigma = 0.25;
    double muJ = -0.05;
    double sigmaJ = 0.15;
    double lambdaJ = 0.0;

    // Functional parameters for the `CalculateOptionPrice` method
    int M = 100000;
    int N = 100;
    bool isCall = true;

    MertonJumpDiffusion params = {spot, strike, T, r, sigma, muJ, sigmaJ, lambdaJ};

    auto results = CalculateOptionPrice(params, M, N, isCall);

    for (const auto &[key, value] : results)
    {
        std::cout << key << ": " << value << std::endl;
    }

    double bs_price = BlackScholesPrice(spot, strike, r, sigma, T, isCall);

    std::cout << "BSM theoretical price is " << bs_price << std::endl;

    return 0;
}
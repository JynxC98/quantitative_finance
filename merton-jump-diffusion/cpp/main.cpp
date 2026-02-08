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
    std::vector<std::vector<double>> grid(M, std::vector<double>(N, 0.0));

    // Generating Normal distribution
    std::normal_distribution norm(0.0, 1.0);

    // Generating Poission's distribution
    std::poisson_distribution poisson(p.lambdaJ * dt);
    // Populating the grid
    for (int m = 0; m < M; ++m)
    {
        // Calculating the drift of the path.
        // The drift will remain the same throughout the paths.
        auto drift = (p.r - 0.5 * (p.sigma * p.sigma) - p.lambdaJ * k) * dt;

        for (int n = 0; n < N; ++n)
        {

            double previous_spot_log;

            if (n == 0)
            {
                // Storing the log-evolution of the underlying spot
                previous_spot_log = std::log(p.spot);
            }
            else
            {
                previous_spot_log = grid[m][n - 1]; // Fetching the spot price from
                                                    // the previous iteration.
            }

            // Calculating the random variables
            // Generating the Brownian increment.

            double dZ = norm(rng);

            // Calculating the number of Jumps
            int num_jumps = poisson(rng);

            // Calculating the `Y_t`term
            double Y_t = 0.0;

            for (int i = 0; i < num_jumps; ++i)
            {
                // Log-normal jump distribution

                std::normal_distribution dJ(p.muJ, p.sigmaJ);

                double jump = dJ(rng);
                Y_t += jump;
            }

            grid[m][n] = previous_spot_log + drift + p.sigma * std::sqrt(dt) * dZ + Y_t;
        }
    }

    // Fetching the payoff

    double mean = 0.0;
    double M2 = 0.0; // sum of squared deviations

    for (int m = 0; m < M; ++m)
    {
        double final_price = std::exp(grid[m][N - 1]);

        double payoff = isCall
                            ? std::max(final_price - p.strike, 0.0)
                            : std::max(p.strike - final_price, 0.0);

        double delta = payoff - mean;
        mean += delta / (m + 1);
        M2 += delta * (payoff - mean);
    }
    double discounted_mean = mean * std::exp(-p.r * p.T);

    double variance = M2 / (M - 1);
    double std_dev = std::sqrt(variance);
    double std_error = std_dev / std::sqrt(M);

    double left_error = discounted_mean - 1.96 * std_error;
    double right_error = discounted_mean + 1.96 * std_error;

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
    double muJ = 0.05;
    double sigmaJ = 0.15;
    double lambdaJ = 1;

    // Functional parameters for the `CalculateOptionPrice` method
    int M = 5000;
    int N = 5000;
    bool isCall = true;

    MertonJumpDiffusion params = {spot, strike, T, r, sigma, muJ, sigmaJ, lambdaJ};

    auto results = CalculateOptionPrice(params, M, N, isCall);

    for (const auto &[key, value] : results)
    {
        std::cout << key << ": " << value << std::endl;
    }

    return 0;
}
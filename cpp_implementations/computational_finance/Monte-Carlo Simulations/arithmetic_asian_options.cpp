/**
 * @file arithmetic_asian_options.cpp
 * @brief A script to calculate the value of an arithmetic Asian option with a 95% confidence interval.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>

using namespace std;

/**
 * @brief Calculate the price of an arithmetic Asian option.
 * @param spot The current spot price.
 * @param strike The predetermined strike price.
 * @param T Time to maturity (in years).
 * @param sigma The underlying's volatility.
 * @param r Risk-free interest rate.
 * @param isCall `true` for call option, `false` for put option.
 * @param M Number of simulation paths.
 * @param N Number of time steps.
 * @returns A map containing the option price and the 95% confidence interval.
 */
map<string, double> getArithmeticOptionPrice(double spot,
                                             double strike,
                                             double T,
                                             double sigma,
                                             double r,
                                             bool isCall,
                                             int M = 5000,
                                             int N = 5000)
{
    // Time step
    double dt = T / N;

    double sum_price = 0.0;
    double sum_squared_price = 0.0;

    // Random generator with seed
    mt19937 generator(random_device{}());

    // Standard normal distribution
    normal_distribution<double> normal(0.0, 1.0);

    for (int path = 0; path < M; ++path)
    {

        double current_spot = spot;
        double floating_sum = spot;

        for (int t = 0; t < N; ++t)
        {
            double dW = normal(generator);
            current_spot = current_spot * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * dW);
            floating_sum += current_spot;
        }

        double average_price = floating_sum / (N + 1);
        double payoff = isCall
                            ? max(average_price - strike, 0.0)
                            : max(strike - average_price, 0.0);
        sum_price += payoff;
        sum_squared_price += payoff * payoff;
    }

    // Computing the mean and variance
    double mean_price = sum_price / M;
    double variance = (sum_squared_price / M) - (mean_price * mean_price);
    double stddev = sqrt(variance);

    // Calculating the 95% confidence interval
    double z_score = 1.96;
    double margin_of_error = z_score * stddev / sqrt(M);

    map<string, double> results;
    // Discounted mean price
    results["Option Price"] = exp(-r * T) * mean_price;
    results["Confidence Interval Lower Bound"] = exp(-r * T) * (mean_price - margin_of_error);
    results["Confidence Interval Upper Bound"] = exp(-r * T) * (mean_price + margin_of_error);

    return results;
}

int main()
{
    double spot = 100.0;
    double strike = 110.0;
    double sigma = 0.25;
    double T = 1.0;
    double r = 0.045;
    bool isCall = true;

    // Get the option price and confidence interval
    map<string, double> result = getArithmeticOptionPrice(spot, strike, T, sigma, r, isCall);

    // Print the results
    cout << "Option Price: " << result["Option Price"] << endl;
    cout << "95% Confidence Interval: [" << result["Confidence Interval Lower Bound"]
         << ", " << result["Confidence Interval Upper Bound"] << "]" << endl;

    return 0;
}

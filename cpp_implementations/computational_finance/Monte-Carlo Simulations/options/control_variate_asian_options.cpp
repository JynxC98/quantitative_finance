/**
 * @file control_variate_asian_options.cpp
 * @brief A script to calculate the value of an arithmetic Asian options with a 95%
 * confidence interval incorporating a control variate variance reduction technique.
 *
 * @author Harsh Parikh
 * @date 14-03-2024
 */

#include <iostream>
#include "statistics.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>

using namespace std;

/**
 * @brief Calculate the price of an arithmetic Asian option using a control variate.
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
    // Calculating the time step
    double dt = T / N;

    // These variables are used to calculate the standard deviation.
    double sum_price = 0.0;
    double sum_squared_price = 0.0;

    // Random number generator with seed
    mt19937 generator(random_device{}());

    // Standard normal distribution
    normal_distribution<double> normal(0.0, 1.0);

    // This vector will store the final value of the averaged stock path
    vector<double> average_final_price(M, 0.0);

    // This vector will store the payoff of each path
    vector<double> payoff(M, 0.0);

    // Iterating through every Monte-Carlo path
    for (int path = 0; path < M; ++path)
    {
        double current_spot = spot;
        double floating_sum = 0.0;

        // Iterating through every timestep
        for (int t = 0; t < N; ++t)
        {
            double dW = normal(generator);
            current_spot = current_spot * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * dW);
            floating_sum += current_spot;
        }
        // Calculating the average
        double average_price = floating_sum / N;

        // Storing the average price
        average_final_price[path] = average_price;

        // Calculating the payoff based on the option type
        payoff[path] = isCall ? max(average_price - strike, 0.0) : max(strike - average_price, 0.0);
    }

    // Calculating the value of beta for the control variate.
    double beta = calculateCovariance(average_final_price, payoff) / calculateVariance(payoff);

    // Storing the updated payoff
    vector<double> updated_payoff(M, 0.0);

    for (int path = 0; path < M; ++path)
    {
        // Equivalent martingale measure
        updated_payoff[path] = payoff[path] - beta * (average_final_price[path] -
                                                      spot * exp(r * T));
    }

    // Computing Monte-Carlo statistics
    double mean_payoff = calculateMean(updated_payoff);

    double std_dev = sqrt(calculateVariance(updated_payoff));

    // Calculating the Monte-Carlo confidence interval
    double z_score = 1.96;
    double margin_of_error = z_score * std_dev / sqrt(M);

    // Storing the result in the form of a map
    map<string, double> results;
    results["Option Price"] = exp(-r * T) * mean_payoff;
    results["Standard Deviation"] = std_dev;
    results["Confidence Interval Lower Bound"] = exp(-r * T) * (mean_payoff - margin_of_error);
    results["Confidence Interval Upper Bound"] = exp(-r * T) * (mean_payoff + margin_of_error);

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
    cout << " Standard Deviation " << result["Standard Deviation"] << endl;
    cout << "95% Confidence Interval: [" << result["Confidence Interval Lower Bound"]
         << ", " << result["Confidence Interval Upper Bound"] << "]" << endl;

    return 0;
}

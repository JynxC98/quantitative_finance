/**
 * @file arithmetic_asian.cpp
 * @brief Monte Carlo simulation for pricing arithmetic Asian options under
 * the Geometric Brownian Motion (GBM) model with variance reduction.
 *
 * This script estimates the fair value of arithmetic Asian options using
 * a Monte Carlo approach. It supports both call and put options, and
 * includes a variance reduction technique to improve accuracy.
 *
 * @author Harsh Parikh
 * @date 31st May 2025
 */

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

/**
 * @brief Prices an arithmetic Asian option via Monte Carlo simulation
 * under the Geometric Brownian Motion (GBM) framework.
 *
 * @param spot         Spot price of the underlying asset
 * @param strike       Strike price of the option
 * @param sigma        Annualized volatility of the asset
 * @param T            Time to maturity (in years)
 * @param r            Risk-free interest rate
 * @param M            Number of Monte Carlo simulation paths
 * @param N            Number of time steps per path
 * @param option_type  true for call option, false for put option
 * @return             Estimated option price
 */

long double calculatePrice(double spot,
                           double strike,
                           double sigma,
                           double T,
                           double r,
                           int M = 5000,
                           int N = 5000,
                           bool option_type = true)
{
    // Calculating the time discretisation
    double dt = T / N;

    // Initialising the spot grid
}
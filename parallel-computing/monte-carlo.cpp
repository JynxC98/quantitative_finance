/**
 * @file monte-carlo.cpp
 * @brief A script to calculate the value of a vanilla European option using
 * monte-carlo simulation.
 *
 * @author Harsh Parikh
 * @date 2nd September 2025
 */
#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <algorithm>
#include <cmath>

/**
 * @brief Calculates the price of a vanilla European option using Monte Carlo simulation
 *
 * This function employs geometric Brownian motion to simulate stock price paths and
 * estimates the option value through parallel Monte Carlo methods. The implementation
 * uses OpenMP for multi-threaded execution to improve computational performance.
 *
 * @param spot     Initial stock price (S_0)
 * @param strike   Strike price of the option (K)
 * @param r        Risk-free interest rate (annualized, e.g., 0.05 for 5%)
 * @param sigma    Volatility of the underlying asset (annualized, e.g., 0.2 for 20%)
 * @param T        Time to maturity in years (e.g., 1.0 for one year)
 * @param M        Number of Monte Carlo simulation paths
 * @param N        Number of time steps per path for discretization
 * @param isCall   Option type flag: true for call option, false for put option
 *
 * @return Present value of the option (discounted expected payoff)
 */
double MonteCarloEngine(double spot,
                        double strike,
                        double r,
                        double sigma,
                        double T,
                        int M,
                        int N,
                        bool isCall = true)
{
    // Calculating time discretisation
    double dt = T / static_cast<double>(N);

    // Initialize total payoff
    double total_payoff = 0.0;

    // Setting the number of threads for parallel computing
    int max_threads = omp_get_max_threads();
    std::cout << max_threads << std::endl;
    omp_set_num_threads(max_threads);

// Parallel Monte Carlo simulation
#pragma omp parallel reduction(+ : total_payoff)
    {
        // Each thread gets its own random number generator
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::normal_distribution<double> dist(0.0, 1.0);

#pragma omp for
        for (int i = 0; i < M; ++i) // Iterating through the monte-carlo paths
        {
            double current_spot = spot;

            // Simulate the path to maturity
            for (int j = 0; j < N; ++j) // Iterating through the time steps
            {
                double Z = dist(gen);
                double drift = (r - 0.5 * sigma * sigma) * dt;
                double diffusion = sigma * sqrt(dt) * Z;
                current_spot *= exp(drift + diffusion);
            }

            // Calculate payoff for this path
            double path_payoff = isCall ? std::max(current_spot - strike, 0.0)
                                        : std::max(strike - current_spot, 0.0);
            total_payoff += path_payoff;
        }
    }

    // Calculate the discounted average payoff
    double option_price = exp(-r * T) * total_payoff / static_cast<double>(M);

    return option_price;
}

int main()
{
    double spot = 100.0;
    double strike = 100.0;
    double r = 0.035;
    double sigma = 0.25;
    double T = 1.0;
    int M = 5000;
    int N = 5000;

    auto price = MonteCarloEngine(spot, strike, r, sigma, T, M, N);

    std::cout << "The option price is " << price << std::endl;
    return 0;
}
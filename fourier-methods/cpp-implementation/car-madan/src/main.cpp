/**
 * @file main.cpp
 * @brief The main script to price options using Car-Madan FFT technique.
 *
 * @author Harsh Parikh
 * @date 19th July 2025
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "characteristic_functions.hpp"
#include <helper_functions.hpp>
#include "fourier_transform.hpp"

/**
 * @brief Computes the price of a European option using the Carr–Madan FFT method.
 *
 * This function implements the Carr–Madan (1999) framework to efficiently compute
 * option prices via the Fast Fourier Transform (FFT). The method evaluates the
 * damped characteristic function of the log-asset price and recovers option values
 * by applying inverse Fourier techniques. This approach is particularly effective
 * for pricing across a grid of strikes.
 *
 * @param spot     Initial stock price (S₀)
 * @param strike   Target strike price for which the option is priced
 * @param sigma    Volatility of the underlying asset
 * @param r        Risk-free interest rate (annualized)
 * @param T        Time to maturity (in years)
 * @param t        Current time (typically 0.0)
 * @param isCall   Set to true for call options, false for puts (currently not used in FFT logic)
 * @param dv       Frequency domain spacing (Δv); controls integration resolution
 * @param N        Number of grid points (must be a power of 2 for FFT efficiency)
 * @param alpha    Damping factor α > 0 to ensure square-integrability of the payoff transform
 *
 * @return The computed option price for the specified strike.
 */

double CarMadanFourierEngine(double spot,
                             double strike,
                             double sigma,
                             double r,
                             double T,
                             double t = 0.0,
                             bool isCall = true,
                             double dv = 0.25,
                             int N,
                             double alpha = 0.25)
{

    // Creating the grid points for the frequency domain
    std::vector<double> freq_grid_pts(N, 0.0);

    // Populating the grid points
    for (int i = 0; i < N; ++i)
    {
        freq_grid_pts[i] = i * dv;
    }

    // Creating the log-strike spacing

    double dk = 2 * M_PI / ((double)N * dv); // Nyquist-Shanon condition to avoid aliasing
    double b = 0.5 * N * dk;                 // The `b` term in log-strike spacing

    // Computing the log-strike domain
    std::vector<double> log_strike_grid(N, 0.0);

    for (int i = 0; i < N; ++i)
    {
        log_strike_grid[i] = -b + i * dk;
    }

    // Computing the `Psi` grid
    std::vector<std::complex<double>> psi_grid(N);

    std::complex<double> i(0.0, 1.0); // Defining the complex number

    for (int i = 0; i < N; ++i)
    {
        //  Simpson’s Rule integration weight based on index `i`:
        // 1/3 for endpoints (i = 0 or i = N - 1)
        // 2/3 for even-indexed interior points
        // 4/3 for odd-indexed interior points
        double w = (i == 0 || i == N - 1) ? 1.0 / 3 : (i % 2 == 0) ? 2.0 / 3
                                                                   : 4.0 / 3;

        // Fetching the frequency grid point
        auto v = freq_grid_pts[i];

        // Converting the frequency grid point to a complex representation
        std::complex<double> v_mod(v, 0.0); // Modified v

        psi_grid[i] = dv * exp(-r * T) * psi(alpha, v_mod, r, sigma, spot, T) * exp(i * b * v) * w;
    }

    // Feeding the psi_grid to the FFT engine
    auto fourier_rep = discrete_fourier_transform(psi_grid);

    // Fetching the complex representation of the call prices vector
    auto call_prices = inverse_fourier_transform(fourier_rep);

    // Calculating the call prices
    std::vector<double> call_price_output(N);
    std::vector<double> strikes(N);

    for (int j = 0; j < N; ++j)
    {
        double k = log_strike_grid[j]; // log-strike
        strikes[j] = exp(k);           // actual strike value
        call_price_output[j] = (1.0 / M_PI) * exp(-alpha * k) * std::real(call_prices[j]);
    }

    // Calculating the required price of the call option
    auto price = linear_interpolate(strike, strikes, call_price_output);

    return price;
}
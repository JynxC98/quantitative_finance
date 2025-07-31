/**
 * @file main.cpp
 * @brief Option pricing engine using the Carr-Madan Fast Fourier Transform method.
 * This implementation provides a high-performance option pricing framework based on the
 * Carr-Madan (1999) Fourier transform approach. The method leverages characteristic
 * functions and FFT algorithms to efficiently compute option prices across multiple
 * strikes simultaneously
 *
 * @author Harsh Parikh
 * @date 19th July 2025
 */

#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include "characteristic_functions.hpp"
#include "helper_functions.hpp"
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
                             int N,
                             double alpha,
                             double t = 0.0,
                             double dv = 0.3)
{

    // Creating the grid points for the frequency domain
    std::vector<double> freq_grid_pts(N, 0.0);

    // Populating the grid points
    for (int j = 0; j < N; ++j)
    {
        freq_grid_pts[j] = j * dv;
    }

    // Creating the log-strike spacing

    double dk = 2 * M_PI / ((double)N * dv); // Nyquist–Shannon condition to avoid aliasing
    double b = 0.5 * N * dk;                 // The `b` term in log-strike spacing

    // Computing the log-strike domain
    std::vector<double> log_strike_grid(N, 0.0);

    for (int j = 0; j < N; ++j)
    {
        log_strike_grid[j] = -b + j * dk;
    }

    // Computing the `Psi` grid
    std::vector<std::complex<double>> psi_grid(N);

    std::complex<double> i(0.0, 1.0); // Defining the complex number
    for (int j = 0; j < N; ++j)
    {
        // Trapezoidal Rule integration weight based on index `j`:
        double w = (j == 0 || j == N - 1) ? 0.5 : 1.0;

        // Fetching the frequency grid point
        auto v = freq_grid_pts[j];

        // Converting the frequency grid point to a complex representation
        std::complex<double> v_mod(v, 0.0); // Modified v

        psi_grid[j] = exp(-r * (T - t)) * dv * psi(alpha, v_mod, r, sigma, spot, T) * exp(i * b * v) * w;
    }

    // Fetching the complex-valued call prices (via FFT of psi grid)
    auto call_prices = discrete_fourier_transform(psi_grid);

    // Calculating the call prices
    std::vector<double> call_price_output(N);
    std::vector<double> strikes(N);

    for (int j = 0; j < N; ++j)
    {
        double k = log_strike_grid[j]; // log-strike
        strikes[j] = exp(k);           // actual strike value
        call_price_output[j] = (exp(-alpha * k) / (M_PI)) * std::real(call_prices[j]);
    }

    // Calculating the required price of the call option
    auto price = linear_interpolate(strike, strikes, call_price_output);

    return price;
}

int main()
{
    // Option properties
    double spot = 100.0;
    double strike = 110.0;
    double T = 1.0;
    double sigma = 0.25;
    double r = 0.035;

    // Car Madan engine properties
    double alpha = 1;
    double N = 1 << 15; // Using a power of 2 for FFT efficiency

    // Calculating the Carr-Madan price
    double carr_madan_price = CarMadanFourierEngine(spot, strike, sigma, r, T, N, alpha);

    // Calculating the BSM Price
    double bsm_price = BlackScholesPrice(spot, strike, sigma, r, T);

    // Printing the Carr-Madan price
    std::cout << "The Carr-Madan price is " << carr_madan_price << std::endl;

    // Printing the Black-Scholes price
    std::cout << "The Black-Scholes price is " << bsm_price << std::endl;

    // Printing the error
    std::cout << "Pricing error " << std::abs(bsm_price - carr_madan_price) << std::endl;

    return 0;
}
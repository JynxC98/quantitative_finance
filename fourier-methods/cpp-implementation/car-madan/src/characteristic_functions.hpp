/**
 * @file characteristic_functions.hpp
 * @brief Header file defining characteristic functions for various stochastic models.
 *
 * This file provides implementations of characteristic functions commonly used in
 * quantitative finance, including models such as Black-Scholes, Heston, and Variance Gamma.
 * These functions are designed to be directly compatible with Fourier-based option
 * pricing methods, including the Carr–Madan framework.
 *
 * @author Harsh Parikh
 * @date July 8, 2025
 */

#if !defined(CHARACTERISTIC_FUNCTIONS_HPP)
#define CHARACTERISTIC_FUNCTIONS_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

// Initialising a custom template
template <typename T>
using ComplexVec = std::vector<std::complex<T>>;

template <typename T>
using Complex = std::complex<T>;

/**
 * @brief Computes the modified characteristic function used in the Carr–Madan framework.
 *
 * This function returns the dampened (exponentially modified) characteristic function
 * required for Fourier-based option pricing using the Carr–Madan method. The damping
 * factor α ensures integrability of the transformed payoff function in the complex plane.
 *
 * @param alpha Damping (smoothening) factor to ensure square-integrability of the payoff
 * @param u     Complex frequency variable in the Fourier domain
 * @param r     Risk-free interest rate
 * @param sigma Volatility of the underlying asset
 * @param S0    Initial spot price of the underlying
 * @param T     Time to maturity
 * @param t     Current or initial time (typically 0)
 *
 * @return The complex-valued characteristic function evaluated at (u - i·alpha)
 */
template <typename T>
Complex<T> bsm_characteristic_function(double alpha,
                                       double u,
                                       double r,
                                       double sigma,
                                       double S0,
                                       double T,
                                       double t = 0)
{
    // Initialising the complex number
    std::complex<double> i(0.0, 1.0);

    // Calculating the `phi` term of the characteristic function
    Complex<double> phi = exp(i * u * log(S0) + (r - 0.5 * sigma * sigma * (T - t) - 0.5 * sigma * sigma * u * u * (T - t)));

    return phi;
}

#endif

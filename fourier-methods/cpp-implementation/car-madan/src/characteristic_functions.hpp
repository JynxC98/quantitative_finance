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
 * @brief Computes the Black-Scholes characteristic function for log-asset price.
 *
 * This function returns the characteristic function φ(u) = E[e^{iu log(S_T)}]
 * under the risk-neutral Black-Scholes model. It is a core component used in
 * Fourier-based option pricing methods such as Carr–Madan, where φ is evaluated
 * at a complex-shifted argument (u - i(α + 1)) to ensure integrability of the
 * overall integrand — but the characteristic function itself is not modified.
 *
 * @param u     Complex frequency variable (usually shifted externally as u - i(α + 1))
 * @param r     Risk-free interest rate
 * @param sigma Volatility of the underlying asset
 * @param S0    Initial spot price of the underlying
 * @param T     Time to maturity
 * @param t     Initial time (typically 0)
 *
 * @return The complex-valued characteristic function φ(u) of log(S_T)
 */

template <typename T>
Complex<T> bsm_characteristic_function(Complex<T> u,
                                       double r,
                                       double sigma,
                                       double S0,
                                       double T,
                                       double t = 0)
{
    // Initialising the complex number
    Complex<T> i(0.0, 1.0);

    // Calculating the `phi` term of the characteristic function
    Complex<double> phi = exp(i * u * log(S0) + (r - 0.5 * sigma * sigma * (T - t) - 0.5 * sigma * sigma * u * u * (T - t)));

    return phi;
}

/**
 * @brief Computes the Carr–Madan integrand Ψ(u) for Fourier option pricing.
 *
 * This function evaluates the integrand Ψ(u), which is the Fourier transform of the
 * exponentially damped European call price. It is used in the Carr–Madan framework to
 * recover option prices via inverse Fourier transform (typically implemented with FFT).
 *
 * The formula is given by:
 *    Ψ(u) = [e^{-rT} * φ(u - i(α + 1))] / [α^2 + α - u^2 + i(2α + 1)u]
 *
 * where:
 *    - φ(u) is the characteristic function of log(S_T)
 *    - α > 0 is the damping parameter ensuring square-integrability of the payoff
 *    - u is the complex frequency variable in the Fourier domain
 *
 * @param u     Real frequency variable (to be used as complex u in the formula)
 * @param alpha Damping factor (α > 0)
 * @param r     Risk-free interest rate
 * @param T     Time to maturity
 * @param phi   A callable characteristic function: std::function<Complex<T>(Complex<T>)>
 *
 * @return The complex-valued integrand Ψ(u) evaluated at the given frequency
 */
template <typename T>
Complex<T> psi(double alpha,
               Complex<T> u,
               double r,
               double sigma,
               double S0,
               double TT,
               double t = 0)
{
    // Evaluating the characteristic function term
    Complex<T> i(0.0, 1.0);                                                    // Complex number
    Complex<T> u_modified = u - i * (alpha + 1);                               // Modified frequency variable
    auto char_term = bsm_characteristic_function(u_modified, r, sigma, S0, T); // Characteristic function

    // Initialising numerator and denominator terms
    Complex<T> numerator, denominator;

    numerator = exp(-r * TT) * char_term;                          // Calculating the numerator
    denominator = alpha * alpha - u * u + i * (2 * alpha + 1) * u; // Calculating the denominator

    return numerator / denominator;
}

#endif

/**
 * @file solvers.hpp
 * @brief Newton-Raphson solver for Heston model quantile inversion
 *
 * This module implements numerical inversion of the cumulative distribution
 * function (CDF) for the Heston model using Gaussian quadrature and
 * Newton's second-order method (Halley's method) for root finding.
 *
 * @author Harsh Parikh
 */

#if !defined(SOLVERS_HPP)
#define SOLVERS_HPP

#include <iostream>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>

#include "quadrature.hpp"
#include "char_function.hpp"

/**
 * @brief Container for CDF, PDF, and its derivative at a point
 */
struct NewtonMethod
{
    double cdf;   ///< Cumulative distribution function F(x) = P(X ≤ x)
    double pdf;   ///< Probability density function f(x) = F'(x)
    double d_pdf; ///< First derivative of PDF f'(x) = F''(x)
};

/**
 * @brief Numerical integration using Gaussian-Legendre quadrature
 *
 * This function wraps the quadrature module to compute definite integrals
 * over u for the CDF, PDF, or dPDF integrands.
 *
 * @tparam Func Callable type for the integrand function
 * @param function The integrand (calculateCDF, calculatePDF, or calculate_dPDF)
 * @param x The quantile value at which to evaluate the integrand
 * @param p Heston model parameters
 * @return Numerical approximation of the definite integral
 */
template <typename Func>
double calculateIntegral(Func function, double x,
                         const HestonParams &p)
{
    auto func = [&](double u)
    {
        return function(x, u, p);
    };
    double upper = 10.0 / (0.5 * (p.v_t + p.v_u) * p.dt);
    // Finer segments where oscillation is rapid
    std::vector<double> breakpoints = {0.0, 5.0, 20.0, 50.0,
                                       100.0};
    double result = 0.0;
    for (int k = 0; k + 1 < breakpoints.size(); k++)
    {
        result += legendreIntegrate(func, breakpoints[k], breakpoints[k + 1]);
    }
    return result;
}

/**
 * @brief Integrand for computing the cumulative distribution function (CDF)
 *
 * Implements the integrand from Gil-Pelaez inversion formula:
 * F(x) = 1/2 - (1/π) ∫₀^∞ Im(e^{-iux} φ(u))/u du
 *
 * @param x The quantile value at which to evaluate the CDF
 * @param u Integration variable
 * @param p Heston model parameters
 * @return Integrand value -Im(e^{-iux} φ(u))/(u * π)
 */
double CDFIntegrand(double x, double u, const HestonParams &p)
{
    auto phi = CharFunction(p, u);

    if (std::abs(u) < 1e-8)
    {
        // Limit of -Im[e^{-iux}φ(u)]/u as u -> 0
        // Using φ(0)=1 and e^{-iux} ≈ 1 - iux, the term is -Im[(1-iux)(1)]/u = x
        return x / M_PI;
    }

    // Im[e^{-iux}φ(u)] = Im[(cos(ux) - i*sin(ux)) * (Re(φ) + i*Im(φ))]
    // = cos(ux)*Im(φ) - sin(ux)*Re(φ)
    double imag_part = std::cos(u * x) * std::imag(phi) - std::sin(u * x) * std::real(phi);

    return -imag_part / (u * M_PI);
}

/**
 * @brief This function calculates the integral of the CDF integrand.
 */
double calculateCDF(double x, const HestonParams &p)
{
    return 0.5 + calculateIntegral(CDFIntegrand, x, p);
}

/**
 * @brief Integrand for computing the probability density function (PDF)
 *
 * f(x) = (1/π) ∫₀^∞ Re(e^{-iux} φ(u)) du
 */
double PDFIntegrand(double x, double u, const HestonParams &p)
{
    auto phi = CharFunction(p, u);
    // Re[e^{-iux}φ(u)] = Re[(cos(ux) - i*sin(ux)) * (Re(φ) + i*Im(φ))]
    // = cos(ux)*Re(φ) + sin(ux)*Im(φ)
    double real_part = std::cos(u * x) * std::real(phi) + std::sin(u * x) * std::imag(phi);

    return (1.0 / M_PI) * real_part;
}

/**
 * @brief Integrand for computing the first derivative of the PDF (F''(x))
 *
 * d/dx PDF(x) = (1/π) ∫₀^∞ Re(-iu * e^{-iux} φ(u)) du
 * = (u/π) ∫₀^∞ Im(e^{-iux} φ(u)) du
 */
double d_PDFIntegrand(double x, double u, const HestonParams &p)
{
    auto phi = CharFunction(p, u);
    double imag_part = std::cos(u * x) * std::imag(phi) - std::sin(u * x) * std::real(phi);

    return (u / M_PI) * imag_part;
}

/**
 * @brief Newton's second-order (Halley's) method for quantile inversion
 *
 * Solves F(x) - var = 0 using Halley's method for cubic convergence.
 *
 * @param var The target probability [0,1] to invert
 * @param p Heston model parameters
 * @param tolerance Convergence tolerance
 * @param max_iterations Maximum iterations
 * @return Quantile x such that F(x) = var
 */
double runNewtonSolver(double var, const HestonParams &p,
                       double tolerance = 1e-8, int max_iterations = 100)
{
    double x = (p.v_t + p.v_u) / 2 * p.dt; // Trapezoidal method for initial guess

    for (int i = 0; i < max_iterations; ++i)
    {
        double f = calculateCDF(x, p) - var;
        double f_prime = calculateIntegral(PDFIntegrand, x, p);
        double f_double_prime = calculateIntegral(d_PDFIntegrand, x, p);

        // Standard Halley's Update: x_{n+1} = x_n - (2ff') / (2(f')^2 - ff'')
        double numerator = 2.0 * f * f_prime;
        double denominator = 2.0 * std::pow(f_prime, 2) - f * f_double_prime;

        double update = numerator / denominator;
        double x_new = x - update;

        if (std::abs(x_new - x) < tolerance)
        {
            return x_new;
        }

        x = x_new;

        // Safety check to keep x positive for variance distributions
        if (x < 0)
            x = 1e-6;
    }

    throw std::runtime_error("Halley solver did not converge");
}

#endif // SOLVERS_HPP
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
 *
 * This structure holds the three values needed for Newton's optimization
 * method: the cumulative distribution function, probability density function,
 * and the first derivative of the PDF.
 *
 * @note Currently not used in the implementation but kept for potential extensions
 */
struct NewtonMethod
{
    double cdf;   ///< Cumulative distribution function F(x) = P(X ≤ x)
    double pdf;   ///< Probability density function f(x) = F'(x)
    double d_pdf; ///< First derivative of PDF f'(x) = F''(x)
};

/**
 * @brief Integrand for computing the cumulative distribution function (CDF)
 *
 * Implements the integrand from Gil-Pelaez inversion formula:
 * F(x) = 1/2 + (1/π) ∫₀^∞ Im(e^{-iux} φ(u))/u du
 *
 * The imaginary part simplifies to sin(u*x)/u * Re(φ(u))
 *
 * @param x The quantile value at which to evaluate the CDF
 * @param u Integration variable (characteristic function frequency)
 * @param p Heston model parameters (passed to CharFunction)
 * @return Integrand value (2/π) * sin(u*x)/u * Re(φ(u))
 *
 * @note The characteristic function φ(u) is evaluated inside this function
 * @warning Has singularity at u=0 which is handled by the quadrature method
 */
double calculateCDF(double x, double u, const HestonParams &p)
{
    if (std::abs(u) < 1e-8)
    {
        // Avoid division by zero (limit as u → 0)
        auto phi = CharFunction(p, u);
        return (1.0 / M_PI) * x * std::real(phi);
    }

    auto phi = CharFunction(p, u);

    double real_phi = std::real(phi);
    double imag_phi = std::imag(phi);

    return (1.0 / M_PI) *
           ((std::sin(u * x) * real_phi -
             std::cos(u * x) * imag_phi) /
            u);
}
/**
 * @brief Integrand for computing the probability density function (PDF)
 *
 * The PDF is the derivative of the CDF: f(x) = F'(x)
 * f(x) = (1/π) ∫₀^∞ Re(e^{-iux} φ(u)) du
 *
 * @param x The quantile value at which to evaluate the PDF
 * @param u Integration variable (characteristic function frequency)
 * @param p Heston model parameters (passed to CharFunction)
 * @return Integrand value (2/π) * cos(u*x) * Re(φ(u))
 */
double calculatePDF(double x, double u, const HestonParams &p)
{
    auto phi = CharFunction(p, u);

    double real_phi = std::real(phi);
    double imag_phi = std::imag(phi);

    return (1.0 / M_PI) * ((std::cos(u * x) * real_phi + std::sin(u * x) * imag_phi));
}

/**
 * @brief Integrand for computing the first derivative of the PDF (F''(x))
 *
 * This is the second derivative of the CDF, used in Halley's method
 * to achieve cubic convergence rate.
 *
 * @param x The quantile value at which to evaluate dPDF
 * @param u Integration variable (characteristic function frequency)
 * @param p Heston model parameters (passed to CharFunction)
 * @return Integrand value -(2/π) * u * sin(u*x) * Re(φ(u))
 */
double calculate_dPDF(double x, double u, const HestonParams &p)
{
    auto char_func_val = CharFunction(p, u);
    return (-2.0 * M_1_PI) * u * std::sin(u * x) * std::real(char_func_val);
}

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
 * @param lower_lim Lower integration limit (default: 0)
 * @param upper_lim Upper integration limit (default: 80)
 * @return Numerical approximation of the definite integral
 *
 * @note The integration limits are for the variable u (frequency space)
 * @see laguerreIntegrate() for quadrature implementation details
 */
template <typename Func>
double calculateIntegral(Func function, double x,
                         const HestonParams &p)
{
    auto func = [&](double u)
    {
        return function(x, u, p);
    };

    // Segment the domain
    double result = 0.0;

    result += legendreIntegrate(func, 0.0, 5.0);
    result += legendreIntegrate(func, 5.0, 20.0);
    result += legendreIntegrate(func, 20.0, 80.0);

    return result;
}

/**
 * @brief Newton's second-order (Halley's) method for quantile inversion
 *
 * Solves the equation F(x) = var where F(x) is the Heston model CDF.
 * Uses Halley's method (second-order Newton) for cubic convergence:
 *
 * x_{n+1} = x_n - (2 * f * f') / (2 * f'^2 - f * f'')
 *
 * Where f = F(x) - var, f' = PDF, f'' = dPDF
 *
 * The simplified iteration formula uses the discriminant:
 * sqrt_term = 1 - sqrt(1 - (2 * cdf * d_pdf) / pdf^2)
 * new_val = prev - (pdf / d_pdf) * sqrt_term
 *
 * @param var The target uniform random variable [0,1] to invert
 * @param p Heston model parameters
 * @param tolerance Convergence tolerance for x (default: 1e-8)
 * @param max_iterations Maximum iterations allowed (default: 100)
 * @return Quantile x such that F(x) = var
 *
 * @throws std::runtime_error If solution fails to converge within max_iterations
 *
 * @note Initial guess is 0.25 which works for moderate parameter ranges
 * @warning May fail for extreme parameters; consider adjusting initial guess
 * @see calculateIntegral() for the numerical integration component
 */
double runNewtonSolver(double var, const HestonParams &p,
                       double tolerance = 1e-8, int max_iterations = 100)
{
    double prev = 0.25; ///< Initial guess
    double new_val, cdf_val, pdf_val, d_pdf_val, sqrt_term;

    for (int i = 0; i < max_iterations; ++i)
    {
        // Evaluate the CDF, PDF, and dPDF at current point
        cdf_val = calculateIntegral(calculateCDF, prev, p);
        pdf_val = calculateIntegral(calculatePDF, prev, p);
        d_pdf_val = calculateIntegral(calculate_dPDF, prev, p);

        // Halley's method discriminant
        sqrt_term = (1.0 - std::sqrt(1.0 - (2.0 * cdf_val * d_pdf_val) / (pdf_val * pdf_val)));

        // Newton-Halley update step
        new_val = prev - (pdf_val / d_pdf_val) * sqrt_term;

        // Check convergence
        if (std::abs(prev - new_val) < tolerance)
        {
            return new_val;
        }

        prev = new_val;
    }

    throw std::runtime_error("Newton solver did not converge within maximum iterations");
}

#endif // SOLVERS_HPP
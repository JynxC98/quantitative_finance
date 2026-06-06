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
#include <algorithm>
#include <stdexcept>

#include "quadrature.hpp"
#include "char_function.hpp"
#include "heston_params.hpp"
#include "helpers.hpp"

#define damp 0.0005 // Damping factor, hard coded for the time being

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
 * @brief This function fetches the upper bound for the characteristic function
 * where the absolute value of the characteristic function decays below the required
 * tolerance.
 *
 * @param p: The Heston parameters
 *
 * @returns: The value at u at which char function decays below the required tolerance.
 */
inline double findCriticalfreq(const HestonParams &p, double tolerance = 1e-6)
{

    // Evaluating the u grid
    auto u_grid = getLinspace(0.0, 5000.0, 50);

    for (auto &u : u_grid)
    {

        if (std::abs(CharFunction(p, u)) < tolerance)
        {
            return u;
        }
    }

    return 5000.0; // Return the max value when no convergence is observed.
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
 * @return Numerical approximation of the definite integral
 */
template <typename Func>
inline double calculateIntegral(Func function, double x,
                                const HestonParams &p)
{
    auto func = [&](double u)
    { return function(x, u, p); };

    const static double critical_freq = findCriticalfreq(p);

    // Upper limit where damped CF is negligible
    double upper = critical_freq;

    std::vector<double> breakpoints = getLinspace(0.0, upper, 50.0);
    double result = 0.0;
    for (int k = 0; k + 1 < breakpoints.size(); ++k)
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
inline double CDFIntegrand(double x, double u, const HestonParams &p)
{

    auto phi = CharFunction(p, std::complex<double>(u, damp));

    if (std::abs(u) < 1e-8)
    {
        // Limit of -Im[e^{-iux}φ(u)]/u as u -> 0
        // Using φ(0)=1 and e^{-iux} ≈ 1 - iux, the term is -Im[(1-iux)(1)]/u = x
        return x / M_PI;
    }

    // Calculating the CDF as per the literature
    double integrand = std::exp(-damp * u) * (std::sin(u * x) / u) * std::real(phi);

    return integrand;
}

/**
 * @brief This function calculates the integral of the CDF integrand.
 */
inline double calculateCDF(double x, const HestonParams &p)
{
    return 2.0 * M_1_PI * calculateIntegral(CDFIntegrand, x, p);
}

/**
 * @brief Integrand for computing the probability density function (PDF)
 *
 * f(x) = (1/π) ∫₀^∞ Re(e^{-iux} φ(u)) du
 */
inline double PDFIntegrand(double x, double u, const HestonParams &p)
{
    auto phi = CharFunction(p, std::complex<double>(u, damp));

    double integrand = std::exp(-damp * u) * std::cos(u * x) * std::real(phi);

    return 2.0 * M_1_PI * integrand;
}

/**
 * @brief Integrand for computing the first derivative of the PDF (F''(x))
 */
inline double d_PDFIntegrand(double x, double u, const HestonParams &p)
{
    auto phi = CharFunction(p, std::complex<double>(u, damp));

    double integrand = -std::exp(-damp * u) * std::sin(u * x) * u * std::real(phi);

    return 2.0 * M_1_PI * integrand;
}

inline double calculateUEpsilon(const HestonParams &p)
{
    double mu1 = 0.5 * (p.v_u + p.v_t) * p.dt;
    double var = p.sigma * p.sigma * p.v_u * p.dt * p.dt / 2.0;
    double std1 = std::sqrt(std::max(var, 0.0));
    double u_eps = mu1 + 10.0 * std1; // 10 std devs for safety
    return std::max(u_eps, 1e-8);
}

/**
 * @brief This function is used to cache the values of the CDF and store the values
 * in a grid. Using this interpolation, we can fetch the value of the required x which
 * matches the uniformly distributed variable.
 *
 * @param x_grid: The x_grid at which integral values are calculated
 * @param integral_vals: The corresponding integral values for the x_grid
 * @param U: The uniformly variable value.
 */
inline double fetch_interpolated_value(const double &U, const HestonParams &p)
{
    // Generating a grid of x_values
    // The values of x will always be between 0 and 1.

    double u_eps = calculateUEpsilon(p);
    static const auto x_grid = getLinspace(0.0, u_eps, 101.0); // Need 100 points

    // Storing the char function values of the CDF
    static std::vector<double> integral_vals(x_grid.size(), 0.0);

    for (size_t i = 0; i < x_grid.size(); ++i)
    {
        integral_vals[i] = calculateCDF(x_grid[i], p);
    }

    // For interpolation, we have the following problem statement:
    // Given the CDF value `U`, what is the value of x that corresponds to
    // the CDF val.

    return linear_interpolate(U, integral_vals, x_grid);
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
inline double runNewtonSolver(double var, const HestonParams &p,
                              double tolerance = 1e-4, int max_iterations = 100)
{
    double x = ((p.v_t + p.v_u) / 2) * p.dt; // Trapezoidal method for initial guess

    // // The authors use bisection method when the value of var is close to tails
    if (var < 0.05 || var > 0.95)
    {
        // Bracket the root first
        double lo = 1e-7, hi = 1.0;

        // Expand hi until CDF(hi) > var
        while (calculateCDF(hi, p) < var)
            hi *= 2.0;

        for (int i = 0; i < max_iterations; ++i)
        {
            double mid = 0.5 * (lo + hi);
            double f = calculateCDF(mid, p) - var;

            if (std::abs(f) < tolerance)
                return mid;

            f > 0 ? hi = mid : lo = mid;
        }
        return 0.5 * (lo + hi);
    }

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
/**
 * @brief This script is used to solve the integral of the characteristic
 * function using Gaussian quadrature method.
 */
#if !defined(SOLVERS_HPP)
#define SOLVERS_HPP

#include <iostream>
#include <cmath>
#include <complex>
#include <map>

#include "quadrature.hpp"
#include "char_function.hpp"

/**
 * @brief This structure acts as a container for storing CDF (F(x)), PDF (F'(x))
 * and D_PDF(F''(x)) for Newton's optimisation method
 *
 * @param cdf: The cumulative distribution function.
 * @param pdf: The probability distribution function.
 * @param d_pdf: The derivative of probability distribution function.
 */
struct NewtonMethod
{
    double cdf;
    double pdf;
    double d_pdf;
};

#endif
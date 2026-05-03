/**
 * @brief This function is used to evaluate non-central chi squared distribution
 * for a particular value. More details on the same can be accessed here:
 *
 * https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
 *
 * Author: Harsh Parikh
 */

#if !defined(NON_CENTRAL_CHI_SQD_HPP)
#define NON_CENTRAL_CHI_SQD_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "bessel.hpp"

/**
 * @brief This function is the implementation of the non-central chi-squared
 * distribution using the modified Bessel's function of the first kind. The function
 * returns the value of the PDF at value z with the degrees of freedom `dof` and
 * the non-central parameter `lambda_`.
 *
 * @param z: The value at which the PDF value is needed.
 * @param dof: Degrees of Freedom
 * @param lambda: The non-centrality parameter.
 */
double NonCentralChi2PDF(double z,
                         double dof,
                         double lambda_);

/**
 * @brief This function is used to generate a random number from the
 * non-central chi-squared distribution with degrees of freedom `dof` and value
 * `z` with non-centrality term `lambda_`
 *
 * @param z: The value at which the PDF value is needed.
 * @param dof: Degrees of Freedom
 * @param lambda: The non-centrality parameter.
 */
double SampleNonCentralChi2(double dof,
                            double lambda_);

#endif

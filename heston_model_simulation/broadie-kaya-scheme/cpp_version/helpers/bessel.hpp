/**
 * @brief This script is used to calculate the value of the Bessel's function.
 * More details on the same can be found below:
 *
 * https://en.wikipedia.org/wiki/Bessel_function
 *
 * @author Harsh Parikh
 */
#if !defined(BESSEL_HPP)
#define BESSEL_HPP

#include <iostream>
#include <cmath>
#include <complex>
#include "gamma.hpp"

/**
 * @brief This function is the implementation of the power scheme for the
 * Bessel's function when |z| <= threshold.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param tolerance: The error term used to evaluate the convergence.
 * @param num_iterations: The number of terms for the expansion.
 *
 * @returns The logarithm of the Bessel value function at point z.
 */
std::complex<double> PowerScheme(std::complex<double> z,
                                 double alpha,
                                 double tolerance,
                                 int num_iterations,
                                 bool log_space);

/**
 * @brief This function is the implementation of the asymptotic expansion for
 * the Bessel's function when |z| > threshold.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param tolerance: The error term used to evaluate the convergence.
 * @param num_iterations: The number of terms for the expansion.
 *
 * @returns The logarithm of the Bessel value function at point z.
 *
 * @note The asymptotic series is NOT convergent —
 * it is an asymptotic expansion that eventually diverges for any fixed z.
 *
 * The fix is a minimum-term stopping rule: halt as soon as the magnitude of
 * the current term exceeds that of the previous term, keeping only the
 * partial sum up to the smallest term.
 */
std::complex<double> AsymptoticExpansion(std::complex<double> z,
                                         double alpha,
                                         double tolerance,
                                         int num_iterations,
                                         bool log_space);

/**
 * @brief This function stores calculates the value of the modified Bessel's
 * function for the value of `z`. The function uses power series for the small
 * value of |z| and asymptotic expansion for the large value of |z| depending
 * on the threshold. Mathematically, the Bessel function can have a complex
 * number as its order. However, in the implementation presented by
 * Brodie & Kaya, the order is a real number. Hence, for simplicity,
 * the order of the function in the current implementation is a real number.
 *
 * @param z: The point at which the function needs to be evaluated.
 * @param alpha: Order of the Bessel function
 * @param num_iterations: The upper value for the summation.
 * @param tolerance: The error term used to evaluate the convergence.
 * @param threshold: The threshold at which the method is decided.
 *
 * @returns: The logarithm of the Bessel function at `z` to avoid overflow.
 */
std::complex<double> ModifiedBessel(std::complex<double> z,
                                    double alpha,
                                    int num_iterations,
                                    double tolerance,
                                    double threshold,
                                    bool log_space);

#endif
/**
 * @brief This script stores the characteristic function of the conditional
 * variance for the Heston Model simulation.
 *
 * @author Harsh Parikh
 */
#if !defined(CHAR_FUNCTION_HPP)
#define CHAR_FUNCTION_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

#include "bessel.hpp"
#include "heston_params.hpp"

/**
 * @brief The characteristic function of the conditional variance. The formulation
 * of the characteristic function is given as follows:
 *
 * \Phi(a) = E[exp(ia \int_u^t v_s ds) | v_u, v_t]
 *
 * The goal is to find the area under the curve between the points u and t (t > u)
 * for the stochastic evolution of v_t
 *
 * @param p: The Heston parameters defined as the structure above.
 * @param u: The characteristic variable
 */
std::complex<double> CharFunction(const HestonParams &p,
                                  std::complex<double> u);

/**
 * @brief CF evaluated at a real argument
 *
 * Promotes u to complex<double> and forwards to the primary overload.
 * Used in unit tests and for the φ(0) = 1 sanity check.
 *
 * @param p  Heston model parameters
 * @param u  Real CF argument
 * @return   Φ(u) ∈ ℂ
 */
std::complex<double> CharFunction(const HestonParams &p, double u);

#endif
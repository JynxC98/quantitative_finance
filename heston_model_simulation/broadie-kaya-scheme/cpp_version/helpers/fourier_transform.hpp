/**
 * @file fourier_transform.hpp
 * @brief Numerical Fourier Transform for Broadie-Kaya Exact Simulation
 *
 * This header implements FFT-based inversion of the characteristic function
 * to compute the CDF of the integrated variance conditional on endpoints.
 * The method uses the Gil-Pelaez inversion formula with FFT for efficiency.
 *
 * @author Harsh Parikh
 * @date July 1, 2025
 */

#if !defined(FOURIER_TRANSFORM_HPP)
#define FOURIER_TRANSFORM_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "char_function.hpp"

/**
 * @brief Structure to hold the CDF grid from FFT inversion
 */
struct CDFGrid
{
    std::vector<double> cdf; ///< Cumulative distribution values F(x_j)
    double dx;               ///< Spacing of x-grid
    double x_min;            ///< Minimum x value (typically 0 for integrated variance)
};

using ComplexVec = std::vector<std::complex<double>>;

/**
 * @brief Computes the Discrete Fourier Transform (DFT) using the Cooley-Tukey FFT algorithm
 *
 * This function implements the radix-2 Cooley-Tukey algorithm, which efficiently
 * computes the DFT by recursively breaking the problem into smaller subproblems.
 * Time complexity: O(N log N) where N is the input size.
 *
 * The algorithm is given as:
 * -------------------------------------------------------------------------------------------
 * X0,...,N−1 ← ditfft2(x, N, s):             DFT of (x0, xs, x2s, ..., x(N-1)s):
 *   if N = 1 then
 *       X0 ← x0                              trivial size-1 DFT base case
 *   else
 *       X0,...,N/2−1 ← ditfft2(x, N/2, 2s)          DFT of even indices
 *       XN/2,...,N−1 ← ditfft2(x+s, N/2, 2s)        DFT of odd indices
 *       for k = 0 to (N/2)-1 do                      combine DFTs of two halves:
 *           p ← Xk
 *           q ← exp(−2πi/N k) * Xk+N/2
 *           Xk ← p + q
 *           Xk+N/2 ← p − q
 *       end for
 *   end if
 * -------------------------------------------------------------------------------------------
 *
 * Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
 *
 * For more information on FFT, check out:
 * 1. 3B1B Fourier transform: https://www.youtube.com/watch?v=spUNpyF58BY
 * 2. The Fourier transform and its applications [Prof. Brad Osgood]
 * 3. Reducible's FFT explanation: https://www.youtube.com/watch?v=h7apO7q16V0
 *
 * @param input A vector of complex numbers representing the time-domain signal.
 *              Size must be a power of two.
 * @return A vector of complex numbers representing the frequency-domain transform.
 *
 * @note Assumes input size is a power of two. If not, behavior is undefined.
 */
ComplexVec discrete_fourier_transform(const ComplexVec &input);

/**
 * @brief Recursive implementation of the inverse Fast Fourier Transform (IFFT)
 *
 * This function computes the inverse DFT by applying the forward FFT
 * to the complex conjugate of the input, then conjugating the result.
 *
 * Mathematically, for a vector X of size N, the inverse DFT is:
 * x_n = (1/N) * Σ_{k=0}^{N-1} X_k * exp(2πi k n / N)
 *
 * The algorithm exploits the relationship:
 * IFFT(X) = (1/N) * conj(FFT(conj(X)))
 *
 * @param input Frequency-domain signal (size must be power of two)
 * @return Time-domain signal (unscaled, without 1/N factor)
 *
 * @note This is an internal helper function. Use inverse_fourier_transform() instead.
 */
ComplexVec ifft_recursive(const ComplexVec &input);

/**
 * @brief Computes the Inverse Fast Fourier Transform (IFFT) with proper scaling
 *
 * This function wraps the recursive IFFT implementation and applies the
 * necessary normalization factor 1/N to each element, ensuring the result
 * matches the standard IDFT definition.
 *
 * @param input Frequency-domain signal (size must be power of two)
 * @return Time-domain signal with proper 1/N scaling
 *
 * @example
 * ComplexVec freq = {1,0,0,0};  // DC component only
 * ComplexVec time = inverse_fourier_transform(freq);
 * // time = {1/N, 1/N, 1/N, ...}
 */
ComplexVec inverse_fourier_transform(ComplexVec &input);

/**
 * @brief Compute CDF grid via FFT-based Gil-Pelaez inversion
 *
 * This function uses the Gil-Pelaez inversion formula with FFT to efficiently
 * compute the cumulative distribution function of the integrated variance
 * on a uniform grid. The method is:
 *
 * F(x) = 1/2 + (1/π) ∫₀^∞ Im[e^{-iux} φ(u)]/u du
 *
 * The FFT approximates this integral via:
 * a_j = φ(j * du) / j  for j = 1,...,N-1
 * a_0 = 1 (limit as j→0)
 *
 * Then F(x_k) ≈ 1/2 - Im[FFT(a)_k] / π
 *
 * @param p    Heston model parameters (contains v_t, v_u, dt, κ, θ, σ)
 * @param N    Number of grid points (MUST be a power of 2 for FFT)
 * @param du   Frequency grid spacing (step size in characteristic function argument u)
 *
 * @return CDFGrid structure containing:
 *         - cdf: Vector of F(x_j) values on uniform x-grid
 *         - dx:  Spacing of x-grid = 2π / (N * du)
 *         - x_min: Starting x value (typically 0 or near 0)
 *
 * @note The x-grid is defined as x_j = j * dx, where j = 0,1,...,N-1
 * @warning The CDF values are clipped to [0,1] to handle numerical errors
 *
 * @see Gil-Pelaez (1951) "Note on the Inversion Theorem"
 * @see Implementation follows Carr-Madan (1999) FFT approach
 */
CDFGrid computeCDFGrid(const HestonParams &p, int N, double du);

/**
 * @brief Sample integrated variance via inverse transform sampling
 *
 * Given a uniform random variable U ~ Uniform(0,1) and a precomputed CDF grid,
 * this function samples from the distribution of integrated variance using
 * inverse transform sampling with linear interpolation.
 *
 * Algorithm:
 * 1. Binary search to find j such that CDF[j] ≤ U < CDF[j+1]
 * 2. Linear interpolation: x = x_j + (U - CDF[j]) / (CDF[j+1] - CDF[j]) * dx
 *
 * @param U        Uniform random number in [0,1] (e.g., from std::mt19937)
 * @param grid     Precomputed CDF grid from computeCDFGrid()
 * @return         Sample from the integrated variance distribution
 *
 * @pre U must be in [0,1]
 * @pre grid.cdf must be monotonically increasing from ~0 to ~1
 *
 * @note For values outside [0,1], returns clamped extreme values
 * @note Linear interpolation provides O(1/N) accuracy
 *
 * @see Numerical Recipes, Section 7.2 "Transformation Method"
 */
double sampleIntegratedVariance(double U, const CDFGrid &grid);

#endif // FOURIER_TRANSFORM_HPP
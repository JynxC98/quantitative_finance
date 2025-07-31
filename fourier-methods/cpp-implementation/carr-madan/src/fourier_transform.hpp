/**
 * @file fourier_transform.hpp
 * @brief Numerical Fourier Transform for Option Pricing Applications
 *
 * This header defines functions to compute the Fourier transform of a
 * characteristic function using numerical integration techniques.
 * Primarily intended for use in option pricing models such as the
 * Carr–Madan framework, where characteristic functions are transformed
 * to obtain option values efficiently.
 *
 * @ref https://cp-algorithms.com/algebra/fft.html
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

// Initialising a custom template
template <typename T>
using ComplexVec = std::vector<std::complex<T>>;
/**
 * @brief Computes the Discrete Fourier Transform (DFT) of a complex-valued input
 *        sequence using the Cooley-Tukey Fast Fourier Transform (FFT) algorithm.
 *
 * This function implements the radix-2 Cooley-Tukey algorithm, which efficiently
 * computes the DFT by recursively breaking the problem into smaller subproblems.
 * It assumes the input size \( N \) is a power of two and decomposes the DFT
 * into smaller DFTs of size \( N/2 \), reducing the time complexity from O(N^2)
 * to O(N log N).
 * The algorithm is given as:
 * -------------------------------------------------------------------------------------------
 * X0,...,N−1 ← ditfft2(x, N, s):             DFT of (x0, xs, x2s, ..., x(N-1)s):
    if N = 1 then
        X0 ← x0                                     trivial size-1 DFT base case
    else
        X0,...,N/2−1 ← ditfft2(x, N/2, 2s)             DFT of (x0, x2s, x4s, ..., x(N-2)s)
        XN/2,...,N−1 ← ditfft2(x+s, N/2, 2s)           DFT of (xs, xs+2s, xs+4s, ..., x(N-1)s)
        for k = 0 to (N/2)-1 do                      combine DFTs of two halves:
            p ← Xk
            q ← exp(−2πi/N k) Xk+N/2
            Xk ← p + q
            Xk+N/2 ← p − q
        end for
    end if
 * -------------------------------------------------------------------------------------------
 * Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
 *
 * For more information on FFT, check out:
 * 1. 3B1B Fourier transform: https://www.youtube.com/watch?v=spUNpyF58BY
 * 2. The Fourier transform and its applications [Prof. Brad Osgood]
 * 3. Reducible's FFT explanation: https://www.youtube.com/watch?v=h7apO7q16V0
 *
 * @note This function assumes that the size of the input is a power of two.
 *
 *
 * @param input A vector of complex numbers representing the time-domain signal.
 * @return A vector of complex numbers representing the frequency-domain transform.
 */
template <typename T>
ComplexVec<T> discrete_fourier_transform(const ComplexVec<T> &input)
{
    // Checking the number of elements in the input vector
    int n = input.size();

    // Base case of 0
    if (n == 0)
    {
        throw std::invalid_argument("Input to DFT must contain at least one element.");
    }

    // Case where the
    if (n == 1)
    {
        return input;
    }

    // Splitting into even and odd indices
    std::vector<std::complex<T>>
        even(n / 2),
        odd(n / 2);

    for (int i = 0; i < n / 2; ++i)
    {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }
    auto even_transformed = discrete_fourier_transform(even);
    auto odd_transformed = discrete_fourier_transform(odd);

    // Initialising the grid for the transformed inputs
    std::vector<std::complex<T>> result(n);

    for (int i = 0; i < n / 2; ++i)
    {
        // Evaluating the complex numerator and denominator
        std::complex<double> num(0, 1.0);
        auto frequency = -2 * M_PI * num * static_cast<T>(i) / static_cast<T>(n);

        // Calculating the nth root of unity
        auto omega = exp(frequency);

        result[i] = even_transformed[i] + omega * odd_transformed[i];
        result[i + n / 2] = even_transformed[i] - omega * odd_transformed[i];
    }
    return result;
}

/**
 * @brief Computes the Inverse Discrete Fourier Transform (IDFT) of a complex-valued input
 *        sequence using the Cooley-Tukey Fast Fourier Transform (FFT) algorithm.
 *
 * This function implements the radix-2 Cooley-Tukey algorithm to compute the inverse DFT
 * efficiently by exploiting the symmetry of the Fourier matrix. The inverse transform is
 * achieved by computing the forward FFT with the conjugated input, then conjugating the
 * output and scaling by \( 1/N \).
 *
 * It assumes the input size \( N \) is a power of two and recursively breaks the problem
 * into smaller inverse DFTs of size \( N/2 \), preserving the \( O(N \log N) \) time complexity.
 *
 * The algorithm is conceptually:
 * -------------------------------------------------------------------------------------------
 * ifft(x):
 *     1. Conjugate the input vector: \( x^* \)
 *     2. Compute forward FFT on \( x^* \)
 *     3. Conjugate the result: \( X^* \)
 *     4. Scale the output by \( 1/N \)
 * -------------------------------------------------------------------------------------------
 *
 * Mathematically, for a vector \( X \) of size \( N \), the inverse DFT is given by:
 * \[
 * x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{2\pi i k n / N}
 * \]
 *
 * @note This function assumes that the size of the input is a power of two.
 *       It returns the time-domain signal corresponding to a given frequency-domain input.
 *
 * @param input A vector of complex numbers representing the frequency-domain spectrum.
 * @return A vector of complex numbers representing the reconstructed time-domain signal.
 */
template <typename T>

ComplexVec<T>
ifft_recursive(const ComplexVec<T> &input)
{
    // Checking the number of elements in the input vector
    int n = input.size();

    // Base case of 0
    if (n == 0)
    {
        throw std::invalid_argument("Input to DFT must contain at least one element.");
    }

    // Case where the
    if (n == 1)
    {
        return input;
    }

    // Splitting into even and odd indices
    std::vector<std::complex<T>>
        even(n / 2),
        odd(n / 2);

    for (int i = 0; i < n / 2; ++i)
    {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }
    auto even_transformed = ifft_recursive(even);
    auto odd_transformed = ifft_recursive(odd);

    // Initialising the grid for the transformed inputs
    std::vector<std::complex<T>> result(n);

    for (int i = 0; i < n / 2; ++i)
    {
        // Evaluating the complex numerator and denominator
        std::complex<double> num(0, 1.0);
        auto frequency = (2 * M_PI * num * static_cast<T>(i)) / static_cast<T>(n);

        // Calculating the root of nth degree of unity
        auto omega = exp(frequency);

        // The results are required to be scaled
        result[i] = (even_transformed[i] + omega * odd_transformed[i]);
        result[i + n / 2] = (even_transformed[i] - omega * odd_transformed[i]);
    }

    return result;
}

/**
 * @brief Computes the Inverse Discrete Fourier Transform (IDFT) by scaling the output
 *        of the recursive inverse FFT.
 *
 * This function wraps the `ifft_recursive` implementation and applies the necessary
 * normalization factor \( 1/N \) to each element of the result, ensuring that the
 * inverse transform adheres to the standard IDFT definition.
 *
 * @param input A vector of complex numbers representing the frequency-domain signal.
 * @return A vector of complex numbers representing the scaled time-domain reconstruction.
 */

template <typename T>
ComplexVec<T> inverse_fourier_transform(ComplexVec<T> &input)
{
    // Fetching the time-domain signal

    auto result = ifft_recursive(input);
    int n = result.size();
    for (auto &element : result)
    {
        element /= static_cast<T>(n);
    }
    return result;
}

#endif

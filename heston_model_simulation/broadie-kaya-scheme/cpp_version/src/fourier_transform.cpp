/**
 * @brief Implementation of FFT for inverting characteristic function.
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "../helpers/fourier_transform.hpp"

ComplexVec discrete_fourier_transform(const ComplexVec &input)
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
    std::vector<std::complex<double>>
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
    std::vector<std::complex<double>> result(n);

    for (int i = 0; i < n / 2; ++i)
    {
        // Evaluating the complex numerator and denominator
        std::complex<double> num(0, 1.0);
        auto frequency = -2 * M_PI * num * static_cast<double>(i) / static_cast<double>(n);

        // Calculating the nth root of unity
        auto omega = exp(frequency);

        result[i] = even_transformed[i] + omega * odd_transformed[i];
        result[i + n / 2] = even_transformed[i] - omega * odd_transformed[i];
    }
    return result;
}

ComplexVec ifft_recursive(const ComplexVec &input)
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
    std::vector<std::complex<double>>
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
    std::vector<std::complex<double>> result(n);

    for (int i = 0; i < n / 2; ++i)
    {
        // Evaluating the complex numerator and denominator
        std::complex<double> num(0, 1.0);
        auto frequency = (2 * M_PI * num * static_cast<double>(i)) / static_cast<double>(n);

        // Calculating the root of nth degree of unity
        auto omega = exp(frequency);

        // The results are required to be scaled
        result[i] = (even_transformed[i] + omega * odd_transformed[i]);
        result[i + n / 2] = (even_transformed[i] - omega * odd_transformed[i]);
    }

    return result;
}

ComplexVec inverse_fourier_transform(ComplexVec &input)
{
    // Fetching the time-domain signal

    auto result = ifft_recursive(input);
    int n = result.size();
    for (auto &element : result)
    {
        element /= static_cast<double>(n);
    }
    return result;
}

/**
 * @brief Compute CDF grid via FFT-based Gil-Pelaez inversion
 */
CDFGrid computeCDFGrid(const HestonParams &p, int N, double du)
{
    // x-grid spacing from Nyquist theorem
    double dx = 2.0 * M_PI / (N * du);

    // Build input sequence a_j = φ(j*du) / j for j = 0,...,N-1
    ComplexVec a(N);

    // j = 0: limit of φ(u)/u as u→0 is i * E[integral]

    a[0] = std::complex<double>(0.0, 0.0);

    for (int j = 1; j < N; ++j)
    {
        double u = j * du;
        std::complex<double> phi = CharFunction(p, u);

        // Gil-Pelaez integrand: Im[e^{-iux}φ(u)]/(uπ)
        // FFT method uses: a_j = φ(j*du) / j
        a[j] = phi / std::complex<double>(j, 0.0);
    }

    // Apply FFT
    ComplexVec transformed = discrete_fourier_transform(a);

    // Extract CDF values at x_k = k * dx
    CDFGrid grid;
    grid.cdf.resize(N);
    grid.dx = dx;
    grid.x_min = 0.0; // Integrated variance is non-negative

    for (int k = 0; k < N; ++k)
    {
        // Gil-Pelaez formula via FFT: F(x_k) = 0.5 - Im[FFT(a)_k] / π
        double cdf_val = 0.5 - std::imag(transformed[k]) / M_PI;

        // Clamp to [0,1] for numerical stability
        grid.cdf[k] = std::max(0.0, std::min(1.0, cdf_val));
    }

    return grid;
}

/**
 * @brief Sample integrated variance via inverse transform sampling
 */
double sampleIntegratedVariance(double U, const CDFGrid &grid)
{
    // Handle edge cases
    if (U <= grid.cdf[0])
        return grid.x_min;
    if (U >= grid.cdf.back())
        return grid.x_min + (grid.cdf.size() - 1) * grid.dx;

    // Binary search for interval containing U
    size_t lo = 0, hi = grid.cdf.size() - 1;

    while (hi - lo > 1)
    {
        size_t mid = (lo + hi) / 2;
        if (grid.cdf[mid] <= U)
            lo = mid;
        else
            hi = mid;
    }

    // Linear interpolation within the interval
    double cdf_lo = grid.cdf[lo];
    double cdf_hi = grid.cdf[hi];
    double t = (U - cdf_lo) / (cdf_hi - cdf_lo);

    return grid.x_min + (lo + t) * grid.dx;
}
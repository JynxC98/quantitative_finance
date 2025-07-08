/**
 * @file polynomial_mult.cpp
 *
 * @brief Test function uses to evaluate the implementation of the FFT and IFFT
 * using the concept of polynomial multiplication.
 *
 * @author Harsh Parikh
 * @date 8th July 2025
 */
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

#include "../src/fourier_transform.hpp"
#include "../src/helper_functions.hpp"

/**
 * @brief Tests the FFT-based polynomial multiplication implemented in the `src` folder.
 *
 * This function multiplies two polynomials, represented as vectors of coefficients,
 * using the Fast Fourier Transform (FFT) technique. It serves as a verification tool
 * for the correctness and efficiency of the FFT implementation in the codebase.
 *
 * @param poly_A A vector representing the coefficients of the first polynomial,
 *               in ascending order of degree (i.e., poly_A[i] corresponds to x^i).
 * @param poly_B A vector representing the coefficients of the second polynomial,
 *               also in ascending order of degree.
 *
 * @return A vector containing the coefficients of the resulting polynomial product,
 *         computed via FFT-based convolution.
 */
std::vector<std::complex<double>> multiply_polynomials(const std::vector<double> &poly_A,
                                                       const std::vector<double> &poly_B)
{
    // Calculating the total size
    int result_size = poly_A.size() + poly_B.size() - 1; // -1 accounts for indexing
                                                         // starting at 0.

    // Calculating the next power of two (the size of fft must be a power of 2)
    int N = next_power_of_two(result_size);

    // Converting the vectors into complex datatype for function compatibility
    std::vector<std::complex<double>> A_complex(N, 0), B_complex(N, 0);

    // Populating the vector values into their corresponding complex equivalent.

    // Given that the size of complex vector is `N`, some elements would be stored
    // as `0` and the code would function as intended.
    for (int i = 0; i < poly_A.size(); ++i)
    {
        A_complex[i] = std::complex<double>(poly_A[i], 0.0);
    }

    for (int j = 0; j < poly_B.size(); ++j)
    {
        B_complex[j] = std::complex<double>(poly_B[j], 0.0);
    }

    // Converting the vectors into forward FFT representation
    auto A_fft = discrete_fourier_transform(A_complex);
    auto B_fft = discrete_fourier_transform(B_complex);

    // Multiplying the vectors point-wise
    std::vector<std::complex<double>> C_fft(N);

    for (int i = 0; i < N; ++i)
    {
        C_fft[i] = A_fft[i] * B_fft[i];
    }

    // Calculating the inverse of the product of two vectors
    auto C_time = inverse_fourier_transform(C_fft);

    return C_time;
}

int main()
{
    // Initialising dummy vectors
    std::vector<double> poly_A = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<double> poly_B = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    int result_size = poly_A.size() + poly_B.size() - 1;

    // Inputing the dummy vectors into the function
    auto result = multiply_polynomials(poly_A, poly_B);

    // Printing the result (real-part)
    std::cout << "Resultant Coefficients: \n";
    for (int i = 0; i < result_size; ++i) // The iteration will only be carried
                                          // till the result size as the rest
                                          // of the elements would be 0.
    {
        std::cout << round(result[i].real()) << " ";
    }
    return 0;
}
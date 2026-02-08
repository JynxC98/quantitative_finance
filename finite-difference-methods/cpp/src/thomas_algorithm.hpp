/**
 * @file thomas_algorithm.cpp
 * @brief This file stores the implementation of Thomas Algorithm, a tri-diagonal
 * matrix solver to solve the system of linear equations. The functionality of the
 * solver is presented in the docstring of the `thomas_algorithm` method.
 *
 * @author Harsh Parikh
 * @date 11th October 2025
 */
#if !defined(THOMAS_ALGORITHM_HPP)
#define THOMAS_ALGORITHM_HPP

#include <iostream>
#include <vector>

/**
 * The code is referenced from this post:
 * @cite https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-tdma-aka-thomas-algorithm-using-python-with-nump
 * @note The tridiagonal system is of the form:
 * @code
 * [b0 c0  0  0  0] [x0]   [d0]
 * [a1 b1 c1  0  0] [x1]   [d1]
 * [ 0 a2 b2 c2  0] [x2] = [d2]
 * [ 0  0 a3 b3 c3] [x3]   [d3]
 * [ 0  0  0 a4 b4] [x4]   [d4]
 * @endcode
 *
 * @param a: Lower diagonal of the tridiagonal matrix (n-1) elements.
 * @param b: Main diagonal of the tridiagonal matrix (n) elements.
 * @param c: Upper diagonal of the tridiagonal matrix (n-1) elements.
 * @param d: Right hand side of the equation (n) elements.
 * @param epsilon: A small number introduced to avoid division by zero error.
 *
 * @return
 * The solution vector (n elements)
 * @note Time complexity: O(n), where n is the system size.
 */
template <typename T>
std::vector<T> thomas_algorithm(const std::vector<T> &a,
                                const std::vector<T> &b,
                                const std::vector<T> &c,
                                const std::vector<T> &d,
                                double epsilon = 1e-12) // smaller perturbation
{

    // The size of the overall system
    size_t n = d.size();

    // Performing basic sanity checks
    if (n == 0)
        throw std::invalid_argument("Vector b must not be empty");
    // if (a.size() != n - 1 || c.size() != n - 1 || b.size() != n)
    // {
    //     throw std::invalid_argument("Input vector sizes are inconsistent");
    // }

    // Thomas algorithm implementation begins wsith modifying the diagonals
    std::vector<T> c_prime(n - 1, 0.0); // Modified upper diagonal
    std::vector<T> d_prime(n, 0.0);     // Modified right-hand side diagonal
    std::vector<T> x(n, 0.0);           // Pre-defined solution vector of 0s.

    // Forward sweep eliminates the lower diagonal
    T denom0 = b[0];
    if (std::abs(denom0) < epsilon)
        denom0 = (denom0 >= 0 ? epsilon : -epsilon);
    c_prime[0] = c[0] / denom0;
    d_prime[0] = d[0] / denom0;

    // Initializing denominator
    T denominator;
    for (size_t i = 1; i < n; ++i)
    {
        denominator = b[i] - a[i - 1] * c_prime[i - 1];
        if (std::abs(denominator) < epsilon)
            denominator = (denominator >= 0 ? epsilon : -epsilon);

        // c_prime is only defined up to n-2; the last entry is not used in back-substitution
        if (i < n - 1)
            c_prime[i] = c[i] / denominator;
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denominator;
    }

    // Backward substitution for finding x
    x[n - 1] = d_prime[n - 1];
    for (int i = static_cast<int>(n) - 2; i >= 0; --i)
    {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    return x;
}

#endif

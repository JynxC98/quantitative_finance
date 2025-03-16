/**
 * @file thomas_algorthim.cpp
 *
 * @brief A script for implementation of `Thomas Algorithm` for solving a system
 * of tridiagonal matrices.
 *
 * @author Harsh Parikh
 * @date 16-03-2025
 */

#include "helper_functions.h"
#include <iostream>
#include <vector>

using namespace std;

/**
 * @file thomas_algorithm.cpp
 * @brief Implementation of the Thomas algorithm for solving a tridiagonal system.
 *
 * This code is referenced from the following post:
 * https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-tdma-aka-thomas-algorithm-using-python-with-nump
 *
 * The function solves a tridiagonal system using the Thomas algorithm.
 *
 * @param a Lower diagonal of the tridiagonal matrix (n-1 elements)
 * @param b Main diagonal of the tridiagonal matrix (n elements)
 * @param c Upper diagonal of the tridiagonal matrix (n-1 elements)
 * @param d Right-hand side of the equation (n elements)
 * @param epsilon A small value to avoid division by zero error
 *
 * @return A vector representing the solution of the system (n elements)
 *
 * @note The tridiagonal system is of the form:
 *
 *     | b0  c0   0   0   0  |   | x0 |   | d0 |
 *     | a1  b1  c1   0   0  |   | x1 |   | d1 |
 *     |  0  a2  b2  c2   0  | * | x2 | = | d2 |
 *     |  0   0  a3  b3  c3  |   | x3 |   | d3 |
 *     |  0   0   0  a4  b4  |   | x4 |   | d4 |
 *
 */

vector<double> thomasAlgorithm(const vector<double> &a,
                               const vector<double> &b,
                               const vector<double> &c,
                               const vector<double> &d,
                               double epsilon = 1e-8)
{

    // Calculating the number of elements
    auto n = d.size();

    // Modified upper diagonal
    vector<double> c_prime(n - 1, 0.0);

    // Modified right-hand side
    vector<double> d_prime(n, 0.0);

    // Forward sweep: Eliminate the lower diagonal
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n - 1; ++i)
    {
        double denominator = b[i] - a[i] * c_prime[i] + epsilon;
        c_prime[i] = c[i] / denominator;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denominator;
    }

    d_prime[n - 1] = (d[n - 1] - a[n - 1] * d_prime[n - 2]) / (b[n - 1] - a[n - 1] * c_prime[n - 2]);

    // Backward substitution: Solving for x

    vector<double> x(n, 0.0);
    x[n - 1] = d_prime[n - 1];

    for (int i = n - 2; i >= 0; --i)
    {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    return x;
}
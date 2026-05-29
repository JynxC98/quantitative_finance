/**
 * @brief This function is used to generate the nodes and weights of the
 * Legendre Polynomial function.
 *
 * @author Harsh Parikh
 */

#if !defined(LEGENDRE_NODES_HPP)
#define LEGENDRE_NODES_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

struct LegendreNode
{
    double x; // node
    double w; // weight
};

/**
 * @brief Evaluate Legendre polynomial P_n(x) using recurrence relation
 *
 * Recurrence: (n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)
 * Start: P_0(x) = 1, P_1(x) = x
 */
inline double legendrePolynomial(int n, double x)
{
    if (n == 0)
        return 1.0;
    if (n == 1)
        return x;

    double p_prev = 1.0; // P_{n-2}
    double p_curr = x;   // P_{n-1}

    for (int k = 1; k < n; k++)
    {
        double p_next = ((2.0 * k + 1.0) * x * p_curr - k * p_prev) / (k + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    return p_curr;
}

/**
 * @brief Evaluate derivative of Legendre polynomial P'_n(x)
 *
 * Formula: P'_n(x) = n * (x*P_n(x) - P_{n-1}(x)) / (x^2 - 1)
 * Or use recurrence: (1-x^2)P'_n(x) = n(P_{n-1}(x) - x P_n(x))
 */
inline double legendreDerivative(int n, double x)
{
    if (n == 0)
        return 0.0;
    if (n == 1)
        return 1.0;

    double p_n = legendrePolynomial(n, x);
    double p_n_minus_1 = legendrePolynomial(n - 1, x);

    return n * (x * p_n - p_n_minus_1) / (x * x - 1.0);
}

/**
 * @brief Find root of Legendre polynomial using Newton-Raphson method
 *
 * Initial guess: cos(π * (4i - 1) / (4n + 2)) for i-th root
 */
inline double findLegendreRoot(int n, int i, double tol = 1e-15)
{
    // Initial approximation using cosine formula
    double x = std::cos(M_PI * (4.0 * i - 1.0) / (4.0 * n + 2.0));

    // Newton-Raphson iteration
    for (int iter = 0; iter < 100; iter++)
    {
        double p = legendrePolynomial(n, x);
        double p_prime = legendreDerivative(n, x);

        double dx = p / p_prime;
        x -= dx;

        if (std::abs(dx) < tol)
            break;
    }

    return x;
}

/**
 * @brief Generate Gauss-Legendre nodes and weights for n-point quadrature
 *
 * @param n Number of quadrature points (1 to ~100; higher needs more precision)
 * @return vector of (node, weight) pairs sorted by node
 */
inline std::vector<LegendreNode> generateGaussLegendre(int n)
{
    std::vector<LegendreNode> nodes;
    nodes.reserve(n); // Reserves memory for n values

    // Roots are symmetric: x_i = -x_{n+1-i}
    for (int i = 1; i <= n; i++)
    {
        double x = findLegendreRoot(n, i);

        // Calculate weight using formula: w_i = 2 / ((1-x_i^2) * [P'_n(x_i)]^2)
        double p_prime = legendreDerivative(n, x);
        double weight = 2.0 / ((1.0 - x * x) * p_prime * p_prime);

        nodes.push_back({x, weight});
    }

    return nodes;
}

#endif
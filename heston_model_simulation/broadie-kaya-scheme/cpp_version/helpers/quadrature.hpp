/**
 * @brief A script to calculate the integral of a function using Gauss-Legendre quadrature over a finite interval [a, b].
 * @author Harsh Parikh
 * @date 26th April 2025
 */

#ifndef GAUSS_LEGENDRE_QUADRATURE_HPP
#define GAUSS_LEGENDRE_QUADRATURE_HPP

#include <iostream>
#include <vector>
#include "legendre_nodes.hpp"

using namespace std;

/**
 * @brief Returns a cached 512-point Gauss-Legendre node/weight table.
 *
 * The table is generated exactly once, on first call, and reused for the
 * lifetime of the program. Because this is a non-template inline function,
 * every translation unit shares the same static instance (so the expensive
 * Newton solve over the Legendre roots runs only once total).
 *
 */
inline const std::vector<LegendreNode> &gaussLegendre512()
{
    static const std::vector<LegendreNode> nodes = generateGaussLegendre(512);
    return nodes;
}

/**
 * @brief This function is used to calculate the area under the curve using Gauss-Legendre quadrature
 *
 * @param function: The main function to be integrated
 * @param lower_limit: The lower limit of the integral
 * @param upper_limit: The upper limit of the integral
 */

template <typename Func>
double legendreIntegrate(Func func, double lower_limit, double upper_limit)
{
    const auto &nodes = gaussLegendre512(); // cached 512-point table
    double result = 0.0;

    // Scaling factors for the finite interval [lower_limit, upper_limit]
    double half_length = 0.5 * (upper_limit - lower_limit);
    double center = 0.5 * (upper_limit + lower_limit);

    // Performing the Gauss-Legendre integration
    for (const auto &node : nodes)
    {
        // Mapping the nodes from [-1, 1] to [lower_limit, upper_limit]
        double x_mapped = center + half_length * node.x;

        // Adding the elements
        result += node.w * func(x_mapped);
    }

    // Scaling the result by the half-length of the interval
    return half_length * result;
}

#endif
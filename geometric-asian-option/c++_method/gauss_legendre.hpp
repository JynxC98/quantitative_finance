/**
 * @brief A script to calculate the integral of a function using Gauss-Legendre quadrature over a finite interval [a, b].
 * @author Harsh Parikh
 * @date 26th April 2025
 */

#ifndef GAUSS_LEGENDRE_QUADRATURE_HPP
#define GAUSS_LEGENDRE_QUADRATURE_HPP

#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief A struct for representing the weights and nodes.
 */
struct LegendreNode
{
    double x; // node
    double w; // weight
};

/**
 * @brief A function to store the weights and nodes for 5-point Gauss-Legendre quadrature
 */
vector<LegendreNode> legendreQuadratureTable()
{
    static const vector<LegendreNode> table = {
        {-0.90617984594, 0.23692688505},
        {-0.53846931010, 0.47862867049},
        {0.0, 0.56888888889},
        {0.53846931010, 0.47862867049},
        {0.90617984594, 0.23692688505}};
    return table;
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
    auto nodes = legendreQuadratureTable();
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

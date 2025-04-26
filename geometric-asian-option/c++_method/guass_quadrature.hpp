/**
 * @brief A script to calculate the integral of a function using Guassian quadrature.
 * @author Harsh Parikh
 * @date 19th April 2025
 */

#ifndef GUASS_QUADRATURE_HPP
#include <iostream>
#include <vector>
#include "helper_functions.h"

using namespace std;

/**
 * @brief A struct for representing the weights and nodes.
 */
struct LaguerreNode
{
    double x; // node
    double w; // weight
};

/**
 * @brief A function to store the weights and nodes for Laguerre polynomials
 */
vector<LaguerreNode> laguerreQuadratureTable()
{
    // Five-point Gauss-Laguerre quadrature nodes and weights
    return {
        {0.26356, 0.521756},
        {1.4134, 0.398667},
        {3.59643, 0.0759424},
        {7.08581, 0.00361176},
        {12.6408, 0.0000233701}};
}

/**
 * @brief This function is used to calculate the area under the curve using Laguerre
 * basis functions
 *
 * @param function: The main function to be integrated
 * @param upper_limit: The upper limit of the integral
 * @param lower_limit: The lower limit of the integral
 * @param num_grids: The number of grid points
 * @param num_poly: The number of Laguerre polynomials
 */

template <typename Func>
double laguerreIntegrate(Func originalFunc, double upper_limit, double lower_limit = 0.0)
{
    // For finite range integration using Gauss-Laguerre

    // Define the transformed function: multiply by e^x to cancel out the e^(-x) weight
    auto transformedFunc = [&](double x) -> double
    {
        if (x > upper_limit - lower_limit)
        {
            return 0.0; // Cut off integration beyond upper limit
        }
        return exp(x) * originalFunc(x + lower_limit);
    };

    // Apply Gauss-Laguerre quadrature
    auto nodes = laguerreQuadratureTable();
    double result = 0.0;

    for (const auto &node : nodes)
    {
        result += node.w * transformedFunc(node.x);
    }

    return result;
}
#endif
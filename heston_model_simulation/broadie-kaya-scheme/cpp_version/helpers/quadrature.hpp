/**
 * @brief This script is used for calculating the integral of a function
 * using the Gauss-Laguerre Quadrature method, suitable for integrals
 * over [0, inf). More details:
 *
 * https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
 *
 * Nodes and weights sourced from scipy.special.roots_laguerre(8)
 *
 * @author Harsh Parikh
 */

#if !defined(QUADRATURE_HPP)
#define QUADRATURE_HPP

#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief A struct for representing the nodes and weights.
 */
struct LaguerreNode
{
    double x; // node
    double w; // weight
};

/**
 * @brief 8-point Gauss-Laguerre quadrature nodes and weights.
 * Sourced from scipy.special.roots_laguerre(8)
 *
 * Designed for integrals of the form:
 * ∫₀^∞ e^{-u} g(u) du ≈ Σ wᵢ * g(xᵢ)
 */
std::vector<LaguerreNode> laguerreQuadratureTable()
{
    static const std::vector<LaguerreNode> table = {
        {0.17027963230510099979, 0.36918858934163752992},
        {0.90370177679937991219, 0.41878678081434295608},
        {2.2510866298661306893, 0.17579498663717180570},
        {4.2667001702876587937, 0.03334349226121565152},
        {7.0459054023934656973, 0.00279453623522567252},
        {10.758516010180995224, 0.00009076508773358213},
        {15.740678641278004578, 0.00000084857467162725},
        {22.863131736889264106, 0.00000000104800117487}};
    return table;
}

/**
 * @brief Numerical integration over [0, ∞) using Gauss-Laguerre quadrature.
 *
 * Approximates ∫₀^∞ f(u) du by rewriting as ∫₀^∞ e^{-u} [e^u f(u)] du
 * The e^u correction is applied internally so the caller just passes f(u).
 *
 * @param func The integrand f(u) to integrate from 0 to infinity
 * @return Numerical approximation of ∫₀^∞ f(u) du
 */
template <typename Func>
double laguerreIntegrate(Func func)
{
    auto nodes = laguerreQuadratureTable();
    double result = 0.0;

    for (const auto &node : nodes)
    {
        result += node.w * std::exp(node.x) * func(node.x);
    }

    return result;
}

#endif // QUADRATURE_HPP
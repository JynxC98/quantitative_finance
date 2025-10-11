/**
 * @file pricing_engine.cpp
 *
 * @brief This script acts as a pricing engine to calculate the value of an
 * option using implicit finite difference scheme. The script also calculates the
 * first order greeks and gamma as per Bloomberg's convention. The script can be used
 * to calculate prices for both American and European options.
 *
 * @author Harsh Parikh
 * @date 11th October 2025
 */

#if !defined(PRICING_ENGINE_HPP)
#define PRICING_ENGINE_HPP

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <string>
#include "helper_functions.hpp"
#include "thomas_algorithm.hpp"

/**
 * @class PricingEngine
 * @brief Pricing engine for European and American options using the finite difference method (FDM).
 *
 * This class implements the finite difference method for solving the Black-Scholes PDE
 * to compute option prices and Greeks (delta, gamma, vega, theta, rho). The FDM discretizes
 * the underlying asset price (spatial direction) and time (temporal direction) to transform
 * the continuous PDE into a system of linear equations solvable on a grid.
 *
 * The class supports both European options (which can be solved analytically via boundary
 * conditions) and American options (which require checking for early exercise at each time step).
 *
 * @section Methods
 * - computePrice(): Solves the PDE and returns the option price at the initial spot price.
 * - computeDelta(): Calculates the first derivative of option price with respect to spot price.
 * - computeGamma(): Calculates the second derivative of option price with respect to spot price.
 * - computeTheta(): Calculates the time decay of the option value.
 * - computeVega(): Calculates the sensitivity to changes in implied volatility.
 * - computeRho(): Calculates the sensitivity to changes in the risk-free rate.
 *
 * @param spot: The current spot price
 * @param strike: The strike price
 * @param ttm: The time to maturity in years
 * @param r: Annualised risk-free rate
 * @param sigma: Annualised volatility
 * @param  q: Annualised dividend yield
 * @param M: The number of spatial steps
 * @param N: The number of time steps
 * @param isCall: true for call and false for put
 * @param exercise_type: "european" or "american"
 */
template <typename T>
class PricingEngine
{
private:
    T spot;
    T strike;
    T ttm;
    T r;
    T sigma;
    T q;
    int M;
    int N;
    bool isCall;
    std::string exercise_type;

public:
    PricingEngine(T spot, T strike, T ttm, T r, T sigma, int M, int N, bool isCall, string exercise_type)
    {
        this->spot = spot;
        this->strike = strike;
        this->ttm = ttm;
        this->r = r;
        this->sigma = sigma;
        this->q = q;
        this->M = M;
        this->N = N;
        this->isCall = isCall;
        this->exercise_type = exercise_type;
    }
    /**
     * @brief Constructs the option price grid using the implicit finite difference method.
     *
     * Discretizes the Black-Scholes PDE spatially (spot prices) and temporally (time steps)
     * to build a 2D grid of option values at each node. The grid spans from t=0 (maturity)
     * to t=T (current time), and from S=0 to S=S_max (spot price range).
     *
     * @return The option price grid and the spot values.
     */
    std::pair<std::vector<std::vector<T>>, std::vector<T>> getOptionGrid();
    /**
     * Computes the fair value of the option at the current spot price (t=0).
     *
     * Extracts the option price from the computed grid at the initial spot price
     * by interpolating or looking up the appropriate grid node.
     *
     * @param option_grid: The option price grid returned from `getOptionGrid`.
     * @param spot_vals: The equidistant grid of spot values returned from `getOptionGrid`
     *
     * @return The value of the option.
     */
    T computePrice(const std::vector<vector<T>> &option_grid,
                   const std::vector<T> &spot_vals);

    /**
     * @brief Computes delta: the first derivative of option price with respect to spot price.
     *
     * Delta measures the rate of change of option price as the underlying spot price changes.
     * Calculated as the first central difference of the option price grid.
     *
     * @param option_grid: The option price grid returned from `getOptionGrid`.
     * @param spot_vals: The equidistant grid of spot values returned from `getOptionGrid`
     *
     * @return The delta of the option.
     */
    T computeDelta(const std::vector<vector<T>> &option_grid,
                   const std::vector<T> &spot_vals);

    /**
     * @brief Computes gamma: the second derivative of option price with respect to spot price.
     *
     * Gamma measures the rate of change of delta as the underlying spot price changes.
     * Calculated as the second central difference of the option price grid.
     *
     * @param option_grid: The option price grid returned from `getOptionGrid`.
     * @param spot_vals: The equidistant grid of spot values returned from `getOptionGrid`
     *
     * @return The gamma of the option.
     */
    T computeGamma(const std::vector<vector<T>> &option_grid,
                   const std::vector<T> &spot_vals);
    /**
     * @brief Computes rho: the sensitivity of option price to changes in the risk-free rate.
     *
     * Rho measures the rate of change of option price with respect to the risk-free interest rate.
     * For interest rate sensitivities, computed via finite differences on the option grid
     * or by numerical perturbation of the rate parameter.
     *
     * @param option_grid: The option price grid returned from `getOptionGrid`.
     * @param spot_vals: The equidistant grid of spot values returned from `getOptionGrid`
     *
     * @return The rho of the option.
     */
    T computeRho(const std::vector<vector<T>> &option_grid,
                 const std::vector<T> &spot_vals);

    /**
     * @brief Computes vega: the sensitivity of option price to changes in volatility.
     *
     * Vega measures the rate of change of option price with respect to the underlying asset's
     * implied volatility. Typically computed via numerical perturbation of the sigma parameter
     * and recalculating the option price grid.
     *
     * @param option_grid: The option price grid returned from `getOptionGrid`.
     * @param spot_vals: The equidistant grid of spot values returned from `getOptionGrid`
     *
     * @return The vega of the option.
     */
    T computeVega(const std::vector<vector<T>> &option_grid,
                  const std::vector<T> &spot_vals);
};

template <typename T>
std::pair<std::vector<std::vector<T>>, std::vector<T>> PricingEngine<T>::getOptionGrid()
{
    // Sanity check for option's exercise type

    if (exercise_type != "european" && exercise_type != "american")
    {
        throw std::invalid_argument("Please select one from `european` and `american`");
    }

    // Calculating the time step
    T dt = ttm / static_cast<T>(N);

    // Assuming the max spot price is 200% of the spot
    T spot_max = 2 * spot;

    // Calculating the spatial step
    T dS = spot_max / static_cast<T>(M);

    // Initializing the grid
    std::vector<std::vector<T>> grid(M + 1, std::vector<T>(N + 1, 0.0));

    // Fetching the equidistant spot values
    auto spot_grid = getEquidistantGrid(M, 0, spot_max); // Assuming the minimum spot would be zero

    // Setting up the terminal conditions
    for (int i = 0; i <= M; ++i)
    {
        // Irrespective of American and European exercise type, the payoff condition
        // would be the same for time T.
        auto value = isCall ? std::max(spot_grid[i] - strike, 0.0)
                            : std::max(strike - spot_grid[i], 0.0);
        grid[i][N] = value;
    }
    // Setting up the boundary conditions.
    auto time_grid = getEquidistantGrid(N, 0, ttm); // ensure this returns N+1 points [0..N]

    for (int i = 0; i <= N; ++i)
    {
        auto tau = ttm - time_grid[i]; // This LOC calculates the time difference
        if (isCall)
        {
            // The difference between the payoffs of the European and American
            // option lies in the heart of its payoff nature. Given that the American
            // option can be exercised at any point of time, the discounting of
            // the strike price differs in the payoff equation.

            auto value = (exercise_type == "european")
                             ? spot_max * std::exp(-q * tau) - strike * std::exp(-r * tau)
                             : spot_max - strike; // For American call, immediate exercise value at large S
            grid[M][i] = value;

            // For call option, the value at spot=0 would just be 0
            grid[0][i] = 0.0;
        }
        else
        {
            // For put options, the payoff differ between the European and American
            // options in the context of strike price adjustment. Given that the option
            // can be exercised at any point of time for American, the option value
            // would just be strike.

            auto value = (exercise_type == "european")
                             ? strike * std::exp(-r * tau)
                             : strike; // For American put, immediate exercise value at S=0
            grid[0][i] = value;

            // For put option, the value at a large spot would be 0
            grid[M][i] = 0.0;
        }
    }

    // Precomputing the coefficients for the tridiagonal system
    std::vector<T> a(M - 1, 0.0); // Lower diagonal  (size M-1)
    std::vector<T> b(M - 1, 0.0); // Main diagonal   (size M-1)
    std::vector<T> c(M - 1, 0.0); // Upper diagonal  (size M-1)

    // Avoiding the boundary values for calculating the coefficients
    for (int i = 1; i < M; ++i)
    {
        auto current_spot = spot_grid[i]; // Spot value at index i  // FIX: was spot[i]

        // Coefficient for V_{n+1, i-1}
        a[i - 1] = 0.5 * dt * (sigma * sigma * (current_spot * current_spot / (dS * dS)) - (r - q) * (current_spot / dS));

        // Coefficient for V_{n+1, i}
        b[i - 1] = -(1.0 + dt * (sigma * sigma * (current_spot * current_spot / (dS * dS)) + r));

        // Coefficient for V_{n+1, i+1}
        c[i - 1] = 0.5 * dt * (sigma * sigma * (current_spot * current_spot / (dS * dS)) + (r - q) * (current_spot / dS));
    }

    // Iterating backwards in time from maturity to present
    // This vector will stoer the RHS values
    std::vector<T> rhs(M - 1, 0.0);
    for (int n = N - 1; n >= 0; --n)
    {
        // Setting up the RHS of the linear system
        for (int i = 1; i < M; ++i)
        {
            rhs[i - 1] = -grid[i][n + 1]; // consistent with sign convention in b
        }
        // Adjusting the boundary conditions in the RHS
        // FIX: use rhs (not d), and correct indices (first and last interior rows)
        rhs[0] -= a[0] * grid[0][n];
        rhs[M - 2] -= c[M - 2] * grid[M][n];

        // Using the Thomas algorithm to solve the tridiagonal system
        auto solution = thomas_algorithm(a, b, c, rhs);

        // This loop is to compare the intrinsic value with the present value
        // in the event of an American option exercise type.
        for (int i = 1; i < M; ++i)
        {
            auto tau = ttm - time_grid[n];

            // Calculating the payoff.
            auto payoff = isCall ? std::max(spot_grid[i] - strike, 0.0)
                                 : std::max(strike - spot_grid[i], 0.0);

            // Calculating the value of the option at the given node
            // The maximum of the option's intrinsic value and present value
            auto node_price = (exercise_type == "european")
                                  ? solution[i - 1]
                                  : std::max(payoff, solution[i - 1]);

            grid[i][n] = node_price;
        }
    }
    return {grid, spot_grid};
}

#endif
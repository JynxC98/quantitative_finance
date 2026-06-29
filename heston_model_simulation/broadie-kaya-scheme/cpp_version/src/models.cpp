/**
 * @brief This script stores the overall implementation of the Euler's
 * and Milstein's scheme for simulating the underlying asset path.
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stdlib.h>
#include <filesystem>
#include <string>
#include <fstream>

#include "../helpers/helpers.hpp"
#include "../helpers/heston_params.hpp"
#include "../helpers/models.hpp"
#include "../helpers/integrated_variance.hpp"
#include "../helpers/random_utils.hpp"
#include "../helpers/cdf_table.hpp"

std::pair<StatisticalProperties, StatisticalProperties> EulerScheme(const HestonParams &p,
                                                                    const OptionParams &o,
                                                                    int M,
                                                                    int N,
                                                                    bool isCall = true,
                                                                    VariancePrevention prevention = VariancePrevention::Truncation)
{
    // We only need to store the results at the end path
    std::vector<double> call_prices(M, 0.0);
    std::vector<double> put_prices(M, 0.0);

    // Calculating the value of the timestep

    double dt = o.T / static_cast<double>(N);

    // This variable stores the transformed variance based on the prevention
    // criteria
    double trans_var;

    // These variables store the standard normal variables
    double dW1, dW2;

    for (int i = 0; i < M; ++i)
    {
        // These variables store the evolution of the current spot and variance
        double current_spot = o.spot;   // The spot at t = 0
        double current_variance = p.v0; // The variance at t = 0

        for (int j = 1; j <= N; ++j)
        {
            // Initialzing the random variabbles dW1 and dW2 based on the
            // literature

            dW1 = normal(gen);

            dW2 = p.rho * dW1 + std::sqrt(1.0 - p.rho * p.rho) * normal(gen);

            switch (prevention)
            {

            case VariancePrevention::Reflection:

                trans_var = std::abs(current_variance);
                break;

            case VariancePrevention::Truncation:

                trans_var = std::max(current_variance, 0.0);
                break;
            default:
                throw std::runtime_error("Unexpected value in switch");
            }

            current_spot = current_spot * (1.0 + o.r * dt + std::sqrt(trans_var) * std::sqrt(dt) * dW1);
            current_variance = current_variance + p.kappa * (p.theta - current_variance) * dt + p.sigma * std::sqrt(trans_var) * std::sqrt(dt) * dW2;
        }

        // Storing the evolution of the path

        call_prices[i] = std::exp(-o.r * o.T) * std::max(current_spot - o.strike, 0.0);
        put_prices[i] = std::exp(-o.r * o.T) * std::max(o.strike - current_spot, 0.0);
    }

    auto results_call = calculateStatistics(call_prices);
    auto results_put = calculateStatistics(put_prices);

    return {results_call, results_put};
}

/**
 * @brief Simulates the Heston model using the Broadie-Kaya exact scheme.
 *
 * Precomputes a 2D grid of CDF tables indexed by (v_u, v_t) to accelerate
 * integrated variance sampling. The grid is persisted to disk and reloaded
 * on subsequent calls, skipping the precomputation cost.
 *
 * For (v_u, v_t) pairs outside [v_min, v_max], falls back to the Newton
 * solver for integrated variance sampling.
 *
 * @param p           Heston model parameters.
 * @param o           Option contract parameters.
 * @param M           Number of Monte Carlo paths.
 * @param N           Number of timesteps per path.
 * @param isCall      True for call option, false for put.
 * @param cache_path  Path to binary cache file for CDF table grid.
 *                    If the file exists, tables are loaded from disk.
 *                    If not, tables are built and saved to this path.
 *
 * @return StatisticalProperties of the discounted payoff distribution.
 *
 * @warning The cache is only valid for the exact (p.theta, n_v, n_points)
 *          combination used during construction. Delete the cache file
 *          whenever any of these change.
 */
std::pair<StatisticalProperties, StatisticalProperties> simulateBroadieKayaHeston(const HestonParams &p,
                                                                                  const OptionParams &o,
                                                                                  int M, int N,
                                                                                  bool isCall,
                                                                                  const std::string &cache_path)
{
    int n_v = 50;
    double v_min = 1e-8;
    double v_max = 20.0 * p.theta;

    // Introducing log-spacing for grid refinement.
    auto logspace = [](double lo, double hi, int n)
    {
        std::vector<double> v(n);
        double log_lo = std::log(lo), log_hi = std::log(hi);
        for (int i = 0; i < n; ++i)
            v[i] = std::exp(log_lo + i * (log_hi - log_lo) / (n - 1));
        return v;
    };

    auto v_nodes = getLinspace(1e-12, v_max, n_v);

    std::vector<std::vector<CDFTable>> tables(n_v, std::vector<CDFTable>(n_v));

    if (std::filesystem::exists(cache_path))
    {
        std::cout << "Loading CDF tables from cache." << std::endl;
        loadCDFTableGrid(tables, cache_path);
    }
    else
    {
        HestonParams temp = p;
        for (int i = 0; i < n_v; ++i)
        {
            temp.v_u = v_nodes[i];
            for (int j = 0; j < n_v; ++j)
            {
                temp.v_t = v_nodes[j];
                tables[i][j] = buildCDFTable(temp, 50);
            }
        }
        saveCDFTableGrid(tables, cache_path);
        std::cout << "CDF table precomputation complete. Cache saved." << std::endl;
    }

    // ── Index helpers ─────────────────────────────────────────────────────

    auto clampIndex = [&](double v) -> int
    {
        int idx = static_cast<int>(
            (v - v_min) / (v_max - v_min) * (n_v - 1));
        return std::max(0, std::min(idx, n_v - 1));
    };

    auto getTable = [&](double v_u, double v_t) -> const CDFTable &
    {
        return tables[clampIndex(v_u)][clampIndex(v_t)];
    };

    std::vector<double> call_prices(M, 0.0);
    std::vector<double> put_prices(M, 0.0);

    std::cout << "Starting Simulation" << std::endl;

    for (int path = 0; path < M; ++path)
    {
        HestonParams params = p;
        double S = o.spot;
        params.v_u = params.v0;
        double sum_ST = 0.0;

        for (int step = 0; step < N; ++step)
        {
            double v_t = sampleVt(params, params.v_u);
            params.v_t = v_t;

            double U = uniform(gen);
            double int_var;

            if (v_t < v_min)
            {
                int_var = 0.0;
            }

            else if (v_t >= v_min && v_t <= v_max &&
                     params.v_u >= v_min && params.v_u <= v_max)
            {
                const CDFTable &table = getTable(params.v_u, v_t);
                int_var = sampleFromTable(U, table);
            }
            else
            {
                int_var = runNewtonSolver(U, params);
            }

            S = priceStep(params, S, int_var, o.r);
            params.v_u = v_t;

            sum_ST += S;
        }

        if (path % 1000 == 0)
        {
            std::cout << "Currently on path " << path << std::endl;
            std::cout << "Mean S_T:  " << sum_ST / M << std::endl;
            std::cout << "Expected:  " << o.spot * std::exp(o.r * o.T) << std::endl;
        }

        call_prices[path] = std::exp(-o.r * o.T) * std::max(S - o.strike, 0.0);
        put_prices[path] = std::exp(-o.r * o.T) * std::max(o.strike - S, 0.0);
    }

    return {calculateStatistics(call_prices), calculateStatistics(put_prices)};
}
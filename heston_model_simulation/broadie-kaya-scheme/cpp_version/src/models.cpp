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
                                                                                  int M,
                                                                                  const std::string &cache_path)
{
    int n_v = 20;
    int n_cdf_points = 50; // resolution of each per-cell CDF table (x_grid/cdf_vals length)
    double v_min = 1e-6;
    double v_max = 20.0 * p.theta;

    // Log-spacing for grid refinement: the CIR/non-central-chi-squared
    // density concentrates near v=0 for parameters violating the Feller
    // condition (as here), so a linearly spaced grid under-resolves exactly
    // the region most samples land in.
    auto logspace = [](double lo, double hi, int n)
    {
        std::vector<double> v(n);
        double log_lo = std::log(lo), log_hi = std::log(hi);
        for (int i = 0; i < n; ++i)
            v[i] = std::exp(log_lo + i * (log_hi - log_lo) / (n - 1));
        return v;
    };

    auto v_nodes = logspace(v_min, v_max, n_v);

    // This is a single-step scheme (dt = T, the N-step loop was removed), so
    // v_u is p.v0 for every single path -- it never varies. Discretizing it
    // onto a 20-node log grid (as a 2D v_u x v_t table would) snaps the one
    // value shared by every path onto the nearest node, which can be tens of
    // percent off (e.g. v0=0.010201 snaps to 0.012925, a 26.7% error) and
    // biases every path identically -- a systematic error that more paths
    // cannot average away. Building the table at the exact v_u = p.v0 with
    // only v_t discretized removes that bias entirely, and is 20x cheaper
    // to boot.
    std::vector<std::vector<CDFTable>> tables(1, std::vector<CDFTable>(n_v));

    if (std::filesystem::exists(cache_path))
    {
        std::cout << "Loading CDF tables from cache." << std::endl;
        loadCDFTableGrid(tables, cache_path);
    }
    else
    {
        std::cout << "Computing the Cache" << std::endl;
        HestonParams temp = p;
        temp.v_u = p.v0;
        for (int j = 0; j < n_v; ++j)
        {
            temp.v_t = v_nodes[j];
            tables[0][j] = buildCDFTable(temp, v_min, n_cdf_points);
        }
        saveCDFTableGrid(tables, cache_path);
        std::cout << "CDF table precomputation complete. Cache saved." << std::endl;
    }

    // ── Index helpers ─────────────────────────────────────────────────────
    // v_nodes is log-spaced, so the nearest-node lookup must also work in
    // log-space, or it silently maps back onto a linear (and therefore
    // wrong) node index.

    double log_v_min = std::log(v_min), log_v_max = std::log(v_max);

    auto clampIndex = [&](double v) -> int
    {
        double log_v = std::log(std::max(v, v_min));
        int idx = static_cast<int>(
            (log_v - log_v_min) / (log_v_max - log_v_min) * (n_v - 1) + 0.5);
        return std::max(0, std::min(idx, n_v - 1));
    };

    // v_t is a continuous draw from the noncentral chi-squared transition,
    // so snapping it to the single nearest of only 20 log-spaced table nodes
    // (clampIndex) discards real information -- the true v_t sits between
    // two nodes ~2% of the way off on average. Bracket the two neighboring
    // nodes instead and linearly interpolate the *quantile* (the sampled
    // x for a given U) between them in log(v_t) space; this is exact at the
    // nodes themselves and removes the systematic snap-to-nearest bias in
    // between, at no extra table-build cost.
    auto bracket = [&](double v_t) -> std::pair<int, double>
    {
        double log_v = std::log(std::max(v_t, v_min));
        double pos = (log_v - log_v_min) / (log_v_max - log_v_min) * (n_v - 1);
        int lo = std::max(0, std::min(static_cast<int>(std::floor(pos)), n_v - 2));
        double w = std::max(0.0, std::min(1.0, pos - lo));
        return {lo, w};
    };

    std::vector<double> call_prices(M, 0.0);
    std::vector<double> put_prices(M, 0.0);

    std::cout << "Starting Simulation" << std::endl;

    double spot_new; // For storing updated spot price

    HestonParams current_params = p;

    current_params.v_u = p.v0;

    // v_u is fixed at p.v0 for every path, so this range check only needs to
    // run once rather than being re-evaluated (identically) on every path.
    bool v_u_in_range = (current_params.v_u >= v_min) && (current_params.v_u <= v_max);

    // Antithetic variates: v_t comes from a Poisson-Gamma mixture, which
    // isn't cheaply invertible, so it's drawn once and shared within a pair.
    // U (the integrated-variance quantile) and Z (the terminal log-price
    // normal) are each directly invertible/negatable and are both monotonic
    // drivers of the payoff -- exactly where antithetic pairing (U, 1-U)
    // and (Z, -Z) buys variance reduction. simulatePath resamples only the
    // U/Z-dependent parts for a given shared v_t.
    for (int pair = 0; pair < M; pair += 2)
    {
        double spot_prev = o.spot;

        double v_t = sampleVt(current_params);
        current_params.v_t = v_t;

        double U = uniform(gen);
        double Z = normal(gen);

        auto simulatePath = [&](double U_draw, double Z_draw, int display_path) -> double
        {
            double int_var;
            const CDFTable *table = nullptr;

            if (v_t < v_min)
            {
                int_var = 0.0;
            }
            else if ((v_t >= v_min) && (v_t <= v_max) && v_u_in_range)
            {
                auto [lo, w] = bracket(v_t);
                double x_lo = sampleFromTable(U_draw, tables[0][lo]);
                double x_hi = sampleFromTable(U_draw, tables[0][lo + 1]);
                int_var = (1.0 - w) * x_lo + w * x_hi;
                table = &tables[0][lo];
            }
            else
            {
                std::cout << "Currently on path " << display_path << std::endl;
                std::cout << "The value of v_u " << current_params.v_u << std::endl;
                std::cout << "The value of v_t " << current_params.v_t << std::endl;
                int_var = runNewtonSolver(U_draw, current_params);
            }

            // Guard against pathological quantile-inversion outliers (coarse
            // table interpolation or Newton-solver misconvergence in the
            // tails) that would otherwise send a single path's log-price to
            // +/-inf. calculateUEpsilon gives the same generous (mean +
            // 10*std) bound used to size the CDF table itself, so this only
            // clips samples that are already far outside the distribution's
            // effective support.
            double int_var_bound = calculateUEpsilon(current_params);
            int_var = std::max(0.0, std::min(int_var, int_var_bound));

            double spot = priceStep(current_params, spot_prev, int_var, o.r, Z_draw);
            return spot;
        };

        spot_new = simulatePath(U, Z, pair);

        if (pair % 2000 == 0)
        {
            std::cout << "Currently on path " << pair << std::endl;
            std::cout << "Current S so far: " << spot_new << std::endl;
            std::cout << "Expected:  " << o.spot * std::exp(o.r * o.T) << std::endl;
        }

        call_prices[pair] = std::exp(-o.r * o.T) * std::max(spot_new - o.strike, 0.0);
        put_prices[pair] = std::exp(-o.r * o.T) * std::max(o.strike - spot_new, 0.0);

        if (pair + 1 < M)
        {
            double spot_anti = simulatePath(1.0 - U, -Z, pair + 1);
            call_prices[pair + 1] = std::exp(-o.r * o.T) * std::max(spot_anti - o.strike, 0.0);
            put_prices[pair + 1] = std::exp(-o.r * o.T) * std::max(o.strike - spot_anti, 0.0);
        }
    }

    // Antithetic pairs are negatively correlated by construction, so feeding
    // all M raw prices straight into calculateStatistics (which assumes iid
    // samples) would understate the actual precision gain: the mean comes
    // out the same either way, but the reported std_dev/CI would be the
    // naive iid figure, not the true (tighter) variance of the antithetic
    // estimator. Average each pair first, then compute statistics over the
    // n/2 pair-means -- the correct standard error for antithetic sampling.
    auto pairMeans = [](const std::vector<double> &v)
    {
        std::vector<double> means;
        means.reserve((v.size() + 1) / 2);
        for (size_t k = 0; k + 1 < v.size(); k += 2)
            means.push_back(0.5 * (v[k] + v[k + 1]));
        if (v.size() % 2 == 1)
            means.push_back(v.back());
        return means;
    };

    return {calculateStatistics(pairMeans(call_prices)), calculateStatistics(pairMeans(put_prices))};
}
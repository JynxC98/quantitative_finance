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
#include "../helpers/fourier_implementation.hpp"

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
    // n_v was 20, bottlenecked by cdf_table.hpp::buildCDFTable's per-point
    // adaptive quadrature (~12ms/point, so ~12s just to build 20 tables at
    // 50 points each). E[int_var | v_t] is convex in log(v_t) (verified
    // analytically from the characteristic function), so the linear
    // quantile-interpolation between adjacent v_t nodes used below is a
    // chord across that convex curve -- a real, systematic upward bias
    // that shrinks with the square of the bracket width. Building tables
    // via FFT (fourier_implementation.hpp) instead is ~10,000x cheaper per
    // table, which is what makes a much finer v_t grid affordable: n_v=200
    // shrinks the bracket width ~10x versus n_v=20, which should shrink
    // that convexity bias by roughly two orders of magnitude -- well under
    // the Monte Carlo noise floor at practical path counts -- while still
    // building in a couple of seconds.
    int n_v = 50;
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
        tables[0] = buildCDFTableGridFFT(temp, v_min, v_nodes);
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

    for (int path = 0; path < M; ++path)
    {
        double spot_prev = o.spot;

        double v_t = sampleVt(current_params);
        current_params.v_t = v_t;

        double U = uniform(gen);
        double int_var;

        const CDFTable *table = nullptr;

        if (v_t < v_min)
        {
            int_var = 0.0;
        }

        else if ((v_t >= v_min) && (v_t <= v_max) && v_u_in_range)
        {
            auto [lo, w] = bracket(v_t);
            double x_lo = sampleFromTable(U, tables[0][lo]);
            double x_hi = sampleFromTable(U, tables[0][lo + 1]);
            int_var = (1.0 - w) * x_lo + w * x_hi;
            table = &tables[0][lo];
        }
        else
        {
            std::cout << "Currently on path " << path << std::endl;
            std::cout << "The value of v_u " << current_params.v_u << std::endl;
            std::cout << "The value of v_t " << current_params.v_t << std::endl;
            int_var = runNewtonSolver(U, current_params);
        }

        // Guard against pathological quantile-inversion outliers (coarse
        // table interpolation or Newton-solver misconvergence in the tails)
        // that would otherwise send a single path's log-price to +/-inf.
        // calculateUEpsilon gives the same generous (mean + 10*std) bound
        // used to size the CDF table itself, so this only clips samples
        // that are already far outside the distribution's effective support.
        double int_var_bound = calculateUEpsilon(current_params);
        int_var = std::max(0.0, std::min(int_var, int_var_bound));

        spot_new = priceStep(current_params, spot_prev, int_var, o.r);

        // if (path % 10000 == 0)
        // {
        //     std::cout << "Currently on path " << path << std::endl;
        //     // std::cout << "V_t on path " << path << " is " << v_t << std::endl;
        //     std::cout << "Current S so far: " << spot_new << std::endl;
        //     std::cout << "Expected:  " << o.spot * std::exp(o.r * o.T) << std::endl;
        // }

        call_prices[path] = std::exp(-o.r * o.T) * std::max(spot_new - o.strike, 0.0);
        put_prices[path] = std::exp(-o.r * o.T) * std::max(o.strike - spot_new, 0.0);
    }

    return {calculateStatistics(call_prices), calculateStatistics(put_prices)};
}
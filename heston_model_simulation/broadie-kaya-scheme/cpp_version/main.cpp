/**
 * @brief Convergence study comparing the exact Broadie-Kaya scheme against
 * Euler discretization, reproducing the format of Tables (a)/(b) in
 * Broadie & Kaya (2006), "Exact Simulation of Stochastic Volatility and
 * Other Affine Jump Diffusion Processes."
 *
 * Runs both schemes over a range of Monte Carlo path counts M (10,000 to
 * 100,000 in steps of 15,000), holding the Euler scheme's time-step count
 * fixed at N = 252. The Broadie-Kaya scheme is a single-step exact method
 * (dt = T) and does not use N at all; it is still reported alongside N for
 * a like-for-like table layout, since that is a labeling choice, not a
 * numerical one.
 *
 * Bias, standard error, and RMS error follow the paper's own definitions:
 * Bias = |mean - true_price|, StdError = MC standard error of the mean,
 * RMSError = sqrt(Bias^2 + StdError^2) -- verified against the paper's own
 * published numbers (e.g. Table (b) row 1: sqrt(0.1543^2 + 0.0772^2) =
 * 0.1725, an exact match to their reported RMS error).
 *
 * Prints both tables to the terminal and writes every row to
 * convergence_results.csv.
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include "helpers/heston_params.hpp"
#include "helpers/models.hpp"

namespace
{

    struct RowResult
    {
        std::string scheme;
        int M;
        int N;
        double bias;
        double std_error;
        double rms_error;
        double time_sec;
    };

    RowResult evaluateEuler(const HestonParams &p, const OptionParams &o, int M, int N, double true_price)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [call_result, put_result] = EulerScheme(p, o, M, N, VariancePrevention::Truncation);
        auto t1 = std::chrono::steady_clock::now();

        double bias = std::abs(call_result.mean - true_price);
        double se = call_result.std_dev;
        double rms = std::sqrt(bias * bias + se * se);

        return {"Euler", M, N, bias, se, rms, std::chrono::duration<double>(t1 - t0).count()};
    }

    RowResult evaluateBK(const HestonParams &p, const OptionParams &o, int M, int N, double true_price,
                         const std::string &cache_path)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [call_result, put_result] = simulateBroadieKayaHeston(p, o, M, cache_path);
        auto t1 = std::chrono::steady_clock::now();

        double bias = std::abs(call_result.mean - true_price);
        double se = call_result.std_dev;
        double rms = std::sqrt(bias * bias + se * se);

        return {"Broadie-Kaya", M, N, bias, se, rms, std::chrono::duration<double>(t1 - t0).count()};
    }

    void printTableHeader(const std::string &title)
    {
        std::cout << "\n"
                  << title << "\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << std::left
                  << std::setw(12) << "M"
                  << std::setw(8) << "N"
                  << std::setw(12) << "Bias"
                  << std::setw(14) << "Std Error"
                  << std::setw(12) << "RMS Error"
                  << "Time (sec)\n";
        std::cout << std::string(70, '-') << "\n";
    }

    void printRow(const RowResult &r)
    {
        std::cout << std::left
                  << std::setw(12) << r.M
                  << std::setw(8) << r.N
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << r.bias
                  << std::setw(14) << r.std_error
                  << std::setw(12) << r.rms_error
                  << std::setprecision(2) << r.time_sec << "\n";
    }

}

int main()
{
    // Option/Heston parameters from Broadie & Kaya (2006), Table 1 -- the
    // same benchmark case used throughout this project's tests.
    OptionParams o = {
        .spot = 100.0,
        .strike = 100.0,
        .r = 0.0319,
        .T = 1.0};

    HestonParams p = {
        .kappa = 6.21,
        .theta = 0.019,
        .sigma = 0.61,
        .v_u = 0.010201,
        .v_t = 0.0,
        .dt = o.T, // Broadie-Kaya is a single-step exact scheme: dt = T
        .v0 = 0.010201,
        .rho = -0.70};

    const double true_price = 6.8061;
    const int N_fixed = 252; // trading days/year; used by the Euler scheme only
    const std::string bk_cache_path = "main_cdf_cache.bin";
    const std::string csv_path = "convergence_results.csv";

    std::vector<int> M_values;
    for (int M = 10000; M <= 100000; M += 15000)
        M_values.push_back(M);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================================================================\n";
    std::cout << " Broadie-Kaya Exact Scheme vs. Euler Discretization\n";
    std::cout << " S=" << o.spot << "  K=" << o.strike << "  V0=" << p.v0
              << "  kappa=" << p.kappa << "  theta=" << p.theta
              << "  sigma_v=" << p.sigma << "  rho=" << p.rho << "\n";
    std::cout << " r=" << o.r * 100.0 << "%  T=" << o.T
              << "  true price=" << true_price << "\n";
    std::cout << "================================================================\n";

    // One-time CDF-table construction for the BK scheme. This is a fixed
    // setup cost, shared across every M below (subsequent calls load the
    // cache from disk), not something that scales with M -- so it is timed
    // and reported separately rather than folded into any row's "Time"
    // column, which would otherwise make the M-scaling in table (a) look
    // wrong (a huge, M-independent spike on the first row only).
    std::cout << "\nBuilding Broadie-Kaya CDF table cache (one-time cost)...\n";
    auto warm_t0 = std::chrono::steady_clock::now();
    simulateBroadieKayaHeston(p, o, 100, bk_cache_path);
    auto warm_t1 = std::chrono::steady_clock::now();
    std::cout << "Cache ready in " << std::chrono::duration<double>(warm_t1 - warm_t0).count()
              << " sec.\n";

    std::vector<RowResult> results;

    printTableHeader("(a) Simulation with the exact method (Broadie-Kaya)");
    for (int M : M_values)
    {
        RowResult r = evaluateBK(p, o, M, N_fixed, true_price, bk_cache_path);
        printRow(r);
        results.push_back(r);
    }

    printTableHeader("(b) Simulation with the Euler discretization (N = " + std::to_string(N_fixed) + ")");
    for (int M : M_values)
    {
        RowResult r = evaluateEuler(p, o, M, N_fixed, true_price);
        printRow(r);
        results.push_back(r);
    }

    std::ofstream csv(csv_path);
    csv << "Scheme,M,N,Bias,StdError,RMSError,ComputingTimeSec\n";
    csv << std::fixed << std::setprecision(6);
    for (const auto &r : results)
    {
        csv << r.scheme << "," << r.M << "," << r.N << ","
            << r.bias << "," << r.std_error << "," << r.rms_error << ","
            << r.time_sec << "\n";
    }
    csv.close();

    std::cout << "\nResults written to " << csv_path << "\n";

    return 0;
}

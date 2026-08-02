/**
 * @brief This is the main simulation script for the Euler's method and BK
 * scheme.
 *
 * @author Harsh Parikh
 */
#include <iostream>
#include <cmath>
#include <iomanip>
#include <assert.h>
#include <cassert>
#include <map>
#include <filesystem>

#include "../helpers/heston_params.hpp"
#include "../helpers/models.hpp"

void test_euler()
{
    // Choosing parameters from the paper itself

    int M = 10000; // paths
    int N = 100;   // timesteps

    OptionParams o = {
        .spot = 100.0,
        .strike = 100.0,
        .r = 0.0319,
        .T = 1.0};

    HestonParams p = {
        .kappa = 6.21,
        .theta = 0.019,
        .sigma = 0.61,
        .v_u = 0.04,
        .v_t = 0.04,
        .dt = o.T / static_cast<double>(N),
        .v0 = 0.010201,
        .rho = -0.7};

    std::cout << "\n========== Testing Euler Scheme ==========\n";

    // =====================================================
    // TEST 1: Call price should be positive
    // =====================================================
    auto [call_result, put_result] = EulerScheme(p, o, M, N, VariancePrevention::Truncation);
    auto bsm_call = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, true);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nTest 1: Call price positivity\n";
    std::cout << "  Mean call price : " << call_result.mean << "\n";
    std::cout << "  BSM call price  : " << bsm_call << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  Std deviation   : " << call_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << call_result.left_lc << ", " << call_result.right_lc << "]\n";

    // =====================================================
    // TEST 2: Put price should be positive
    // =====================================================
    auto bsm_put = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, false);

    std::cout << "\nTest 2: Put price positivity\n";
    std::cout << "  Mean put price  : " << put_result.mean << "\n";
    std::cout << "  BSM put price   : " << bsm_put << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(put_result.mean - bsm_put) << "\n";
    std::cout << "  Std deviation   : " << put_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << put_result.left_lc << ", " << put_result.right_lc << "]\n";

    // =====================================================
    // TEST 3: Put-Call Parity
    // C - P = S - K * e^(-rT)
    // =====================================================
    double parity_lhs = call_result.mean - put_result.mean;
    double parity_rhs = o.spot - o.strike * std::exp(-o.r * o.T);
    double parity_error = std::abs(parity_lhs - parity_rhs);
    double bsm_parity_lhs = bsm_call - bsm_put;

    std::cout << "\nTest 3: Put-Call Parity  C - P = S - K*e^(-rT)\n";
    std::cout << "  C - P           : " << parity_lhs << "\n";
    std::cout << "  BSM C - P       : " << bsm_parity_lhs << "\n";
    std::cout << "  S - K*e^(-rT)   : " << parity_rhs << "\n";
    std::cout << "  Error           : " << parity_error << "\n";

    // =====================================================
    // TEST 4: Reflection vs Truncation should be close
    // =====================================================
    auto [refl_result_call, refl_result_put] = EulerScheme(p, o, M, N, VariancePrevention::Reflection);
    double scheme_diff = std::abs(call_result.mean - refl_result_call.mean);

    std::cout << "\nTest 4: Reflection vs Truncation\n";
    std::cout << "  Truncation mean : " << call_result.mean << "\n";
    std::cout << "  Reflection mean : " << refl_result_call.mean << "\n";
    std::cout << "  BSM call price  : " << bsm_call << "\n";
    std::cout << "  Trunc vs BSM    : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  Refl vs BSM    : " << std::abs(refl_result_call.mean - bsm_call) << "\n";
    std::cout << "  Difference      : " << scheme_diff << "\n";
}

void test_BK()
{

    int M = 10000; // paths
    int N = 100;   // timesteps

    double true_price = 6.8061; // The true option (call) price fetched from paper
    // Choosing parameters from the paper itself

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
        .dt = o.T,
        .v0 = 0.010201,
        .rho = -0.7};

    std::cout << "\n========== Testing Broadie-Kaya Scheme ==========\n";

    auto [call_result, put_result] = simulateBroadieKayaHeston(p, o, M);
    auto [euler_call, euler_put] = EulerScheme(p, o, M, N, VariancePrevention::Truncation);

    auto bsm_call = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, true);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nCall price comparision\n";
    std::cout << "  BK Mean call price : " << call_result.mean << "\n";
    std::cout << "  Euler Mean call price : " << euler_call.mean << "\n";
    std::cout << "  Actual vs BK diff  : " << std::abs(true_price - call_result.mean) << "\n";
    std::cout << "  Actual vs Euler diff  : " << std::abs(true_price - euler_call.mean) << "\n";
    std::cout << "  Std deviation   : " << call_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << call_result.left_lc << ", " << call_result.right_lc << "]\n";

    // Testing the put values
    auto bsm_put = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, false);

    std::cout << "\nPut price comparision\n";
    std::cout << "  Mean put price : " << put_result.mean << "\n";
    std::cout << "  BSM put price  : " << bsm_put << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(put_result.mean - bsm_put) << "\n";
    std::cout << "  Std deviation   : " << put_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << put_result.left_lc << ", " << put_result.right_lc << "]\n";

    // =====================================================
    // BK vs Euler should be close(cross - scheme sanity check)
    // =====================================================

    std::cout << "\nBK vs Euler (cross-scheme sanity check)\n";
    std::cout << "  BK mean         : " << put_result.mean << "\n";
    std::cout << "  Euler mean      : " << euler_put.mean << "\n";
}

int main()
{
    // test_euler();
    test_BK();

    return 0;
}
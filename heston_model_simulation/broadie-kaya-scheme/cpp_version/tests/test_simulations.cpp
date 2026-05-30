#include <iostream>
#include <cmath>
#include <iomanip>
#include <assert.h>
#include <cassert>
#include <map>

#include "../helpers/heston_params.hpp"
#include "../helpers/models.hpp"

void test_euler()
{
    // Choosing parameters from the paper itself

    HestonParams p = {
        .kappa = 6.21,
        .theta = 0.019,
        .sigma = 0.61,
        .v_u = 0.04,
        .v_t = 0.04,
        .dt = 1.0 / 252.0,
        .v0 = 0.010201,
        .rho = -0.7};

    OptionParams o = {
        .spot = 100.0,
        .strike = 100.0,
        .r = 0.0319,
        .T = 1.0};

    std::cout << "\n========== Testing Euler Scheme ==========\n";

    int M = 100000; // paths
    int N = 512;    // timesteps

    // =====================================================
    // TEST 1: Call price should be positive
    // =====================================================
    auto call_result = EulerScheme(p, o, M, N, true, VariancePrevention::Truncation);
    auto bsm_call = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, true);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nTest 1: Call price positivity\n";
    std::cout << "  Mean call price : " << call_result.mean << "\n";
    std::cout << "  BSM call price  : " << bsm_call << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  Std deviation   : " << call_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << call_result.left_lc << ", " << call_result.right_lc << "]\n";
    std::cout << "  Test            : " << (call_result.mean > 0.0 ? "✅ PASS" : "❌ FAIL") << "\n";

    // =====================================================
    // TEST 2: Put price should be positive
    // =====================================================
    auto put_result = EulerScheme(p, o, M, N, false, VariancePrevention::Truncation);
    auto bsm_put = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, false);

    std::cout << "\nTest 2: Put price positivity\n";
    std::cout << "  Mean put price  : " << put_result.mean << "\n";
    std::cout << "  BSM put price   : " << bsm_put << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(put_result.mean - bsm_put) << "\n";
    std::cout << "  Std deviation   : " << put_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << put_result.left_lc << ", " << put_result.right_lc << "]\n";
    std::cout << "  Test            : " << (put_result.mean > 0.0 ? "✅ PASS" : "❌ FAIL") << "\n";

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
    std::cout << "  Test            : " << (parity_error < 0.5 ? "✅ PASS" : "❌ FAIL") << "\n";

    // =====================================================
    // TEST 4: Reflection vs Truncation should be close
    // =====================================================
    auto refl_result = EulerScheme(p, o, M, N, true, VariancePrevention::Reflection);
    double scheme_diff = std::abs(call_result.mean - refl_result.mean);

    std::cout << "\nTest 4: Reflection vs Truncation\n";
    std::cout << "  Truncation mean : " << call_result.mean << "\n";
    std::cout << "  Reflection mean : " << refl_result.mean << "\n";
    std::cout << "  BSM call price  : " << bsm_call << "\n";
    std::cout << "  Trunc vs BSM    : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  Refl  vs BSM    : " << std::abs(refl_result.mean - bsm_call) << "\n";
    std::cout << "  Difference      : " << scheme_diff << "\n";
    std::cout << "  Test            : " << (scheme_diff < 1.0 ? "✅ PASS" : "❌ FAIL") << "\n";

    // =====================================================
    // TEST 5: Deep OTM call should be cheaper than ATM
    // =====================================================
    OptionParams o_otm = o;
    o_otm.strike = 150.0;
    auto otm_result = EulerScheme(p, o_otm, M, N, true, VariancePrevention::Truncation);
    auto bsm_otm = BlackScholesPrice(o_otm.spot, o_otm.strike, std::sqrt(p.v0), o_otm.r, o_otm.T, true);

    std::cout << "\nTest 5: ATM call > OTM call\n";
    std::cout << "  ATM call (K=100): " << call_result.mean << "\n";
    std::cout << "  OTM call (K=150): " << otm_result.mean << "\n";
    std::cout << "  BSM ATM  (K=100): " << bsm_call << "\n";
    std::cout << "  BSM OTM  (K=150): " << bsm_otm << "\n";
    std::cout << "  ATM BSM vs MC   : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  OTM BSM vs MC   : " << std::abs(otm_result.mean - bsm_otm) << "\n";
    std::cout << "  Test            : " << (call_result.mean > otm_result.mean ? "✅ PASS" : "❌ FAIL") << "\n";

    std::cout << "\n========== Euler Scheme tests completed ==========\n";
}

void test_BK()
{
    // Choosing parameters from the paper itself
    HestonParams p = {
        .kappa = 6.21,
        .theta = 0.019,
        .sigma = 0.61,
        .v_u = 0.04,
        .v_t = 0.04,
        .dt = 1.0 / 365.0,
        .v0 = 0.010201,
        .rho = -0.7};

    OptionParams o = {
        .spot = 100.0,
        .strike = 100.0,
        .r = 0.0319,
        .T = 1.0};

    std::cout << "\n========== Testing Broadie-Kaya Scheme ==========\n";

    int M = 100000; // paths
    int N = 512;    // timesteps

    // =====================================================
    // TEST 1: Call price should be positive
    // =====================================================
    auto call_result = simulateBroadieKayaHeston(p, o, M, N, true);
    auto bsm_call = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, true);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nTest 1: Call price positivity\n";
    std::cout << "  Mean call price : " << call_result.mean << "\n";
    std::cout << "  BSM call price  : " << bsm_call << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  Std deviation   : " << call_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << call_result.left_lc << ", " << call_result.right_lc << "]\n";
    std::cout << "  Test            : " << (call_result.mean > 0.0 ? "✅ PASS" : "❌ FAIL") << "\n";

    // =====================================================
    // TEST 2: Put price should be positive
    // =====================================================
    auto put_result = simulateBroadieKayaHeston(p, o, M, N, false);
    auto bsm_put = BlackScholesPrice(o.spot, o.strike, std::sqrt(p.v0), o.r, o.T, false);

    std::cout << "\nTest 2: Put price positivity\n";
    std::cout << "  Mean put price  : " << put_result.mean << "\n";
    std::cout << "  BSM put price   : " << bsm_put << "\n";
    std::cout << "  BSM vs MC diff  : " << std::abs(put_result.mean - bsm_put) << "\n";
    std::cout << "  Std deviation   : " << put_result.std_dev << "\n";
    std::cout << "  95% CI          : [" << put_result.left_lc << ", " << put_result.right_lc << "]\n";
    std::cout << "  Test            : " << (put_result.mean > 0.0 ? "✅ PASS" : "❌ FAIL") << "\n";

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
    std::cout << "  Test            : " << (parity_error < 0.5 ? "✅ PASS" : "❌ FAIL") << "\n";

    // =====================================================
    // TEST 4: BK vs Euler should be close (cross-scheme sanity check)
    // =====================================================
    auto euler_call = EulerScheme(p, o, M, N, true, VariancePrevention::Truncation);
    double scheme_diff = std::abs(call_result.mean - euler_call.mean);

    std::cout << "\nTest 4: BK vs Euler (cross-scheme sanity check)\n";
    std::cout << "  BK mean         : " << call_result.mean << "\n";
    std::cout << "  Euler mean      : " << euler_call.mean << "\n";
    std::cout << "  BSM call price  : " << bsm_call << "\n";
    std::cout << "  BK   vs BSM     : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  Euler vs BSM    : " << std::abs(euler_call.mean - bsm_call) << "\n";
    std::cout << "  BK vs Euler     : " << scheme_diff << "\n";
    std::cout << "  Test            : " << (scheme_diff < 1.0 ? "✅ PASS" : "❌ FAIL") << "\n";

    // =====================================================
    // TEST 5: Deep OTM call should be cheaper than ATM
    // =====================================================
    OptionParams o_otm = o;
    o_otm.strike = 150.0;
    auto otm_result = simulateBroadieKayaHeston(p, o_otm, M, N, true);
    auto bsm_otm = BlackScholesPrice(o_otm.spot, o_otm.strike, std::sqrt(p.v0), o_otm.r, o_otm.T, true);

    std::cout << "\nTest 5: ATM call > OTM call\n";
    std::cout << "  ATM call (K=100): " << call_result.mean << "\n";
    std::cout << "  OTM call (K=150): " << otm_result.mean << "\n";
    std::cout << "  BSM ATM  (K=100): " << bsm_call << "\n";
    std::cout << "  BSM OTM  (K=150): " << bsm_otm << "\n";
    std::cout << "  ATM BSM vs MC   : " << std::abs(call_result.mean - bsm_call) << "\n";
    std::cout << "  OTM BSM vs MC   : " << std::abs(otm_result.mean - bsm_otm) << "\n";
    std::cout << "  Test            : " << (call_result.mean > otm_result.mean ? "✅ PASS" : "❌ FAIL") << "\n";

    std::cout << "\n========== Broadie-Kaya Scheme tests completed ==========\n";
}

int main()
{
    // test_euler();
    test_BK();

    return 0;
}
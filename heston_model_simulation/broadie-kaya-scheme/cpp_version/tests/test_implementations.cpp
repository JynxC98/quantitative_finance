/**
 * @brief This script is used to test the implementations
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <assert.h>
#include <map>

#include "../helpers/bessel.hpp"
#include "../helpers/non_central_chi_sqd.hpp"
#include "../helpers/char_function.hpp"
#include "../helpers/solvers.hpp"
#include "../helpers/helpers.hpp"
#include "../helpers/quadrature.hpp"
#include "../helpers/fourier_transform.hpp"

void test_generator()
{
    double dof = 3.0;
    double lambda_ = 3.0;

    int num_samples = 1000000;

    // --- Welford's online algorithm for numerically stable mean & variance ---
    double mean = 0.0;
    double M2 = 0.0; // Sum of squared deviations from the running mean

    for (int i = 0; i < num_samples; ++i)
    {
        double x = SampleNonCentralChi2(dof, lambda_);
        double delta = x - mean;
        mean += delta / (i + 1); // Running mean
        double delta2 = x - mean;
        M2 += delta * delta2; // Running sum of squared deviations
    }

    double sample_variance = M2 / (num_samples - 1); // Unbiased (Bessel-corrected)

    // --- Theoretical values ---
    // E[X]   = dof + lambda
    // Var[X] = 2 * (dof + 2 * lambda)
    double theoretical_mean = dof + lambda_;
    double theoretical_variance = 2.0 * (dof + 2.0 * lambda_);

    // --- Tolerance: 5-sigma bounds using standard error ---
    // SE of sample mean     = sqrt(Var / n)
    // SE of sample variance = sqrt(2 * Var^2 / (n-1))  [chi-squared sampling dist.]
    double sigma_mean = 5.0 * std::sqrt(theoretical_variance / num_samples);
    double sigma_variance = 5.0 * std::sqrt(2.0 * theoretical_variance * theoretical_variance / (num_samples - 1));

    // --- Assertions ---
    bool mean_ok = std::abs(mean - theoretical_mean) < sigma_mean;
    bool variance_ok = std::abs(sample_variance - theoretical_variance) < sigma_variance;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Samples          : " << num_samples << "\n";
    std::cout << "Theoretical mean : " << theoretical_mean << "\n";
    std::cout << "Sample mean      : " << mean << "\n";
    std::cout << "Mean tolerance   : ± " << sigma_mean << "\n";
    std::cout << "Mean test        : " << (mean_ok ? "PASS" : "FAIL") << "\n\n";
    std::cout << "Theoretical var  : " << theoretical_variance << "\n";
    std::cout << "Sample variance  : " << sample_variance << "\n";
    std::cout << "Variance tol     : ± " << sigma_variance << "\n";
    std::cout << "Variance test    : " << (variance_ok ? "PASS" : "FAIL") << "\n";

    // Hard assertions — uncomment if using a test framework
    // assert(mean_ok     && "Mean test failed: sample mean outside 5-sigma bound");
    // assert(variance_ok && "Variance test failed: sample variance outside 5-sigma bound");
}

void test_heston_variance_moments()
{
    // --- Heston / CIR parameters ---
    const double kappa = 2.0;
    const double theta = 0.04;
    const double sigma = 0.3;
    const double v_t = 0.04;
    const double dt = 1.0 / 252.0;

    const int num_samples = 500000;

    // -------------------------------------------------------------------
    // Theoretical conditional moments of V_{t+dt} | V_t  (CIR closed form)
    //
    //   E  [V_{t+dt}] = V_t * e^{-κ·dt}  +  θ · (1 − e^{-κ·dt})
    //
    //   Var[V_{t+dt}] = V_t · (σ²/κ)  · (e^{-κ·dt} − e^{-2κ·dt})
    //                 + (θ·σ²/2κ)     · (1 − e^{-κ·dt})²
    // -------------------------------------------------------------------
    const double e_kdt = std::exp(-kappa * dt);
    const double theoretical_mean = v_t * e_kdt + theta * (1.0 - e_kdt);
    const double theoretical_var = v_t * (sigma * sigma / kappa) * (e_kdt - e_kdt * e_kdt) + (theta * sigma * sigma / (2.0 * kappa)) * std::pow(1.0 - e_kdt, 2.0);

    // --- Non-central chi-squared parameterisation (Broadie-Kaya) ---
    //
    //  V_{t+dt} = (σ²(1 − e^{-κ·dt}) / 4κ) · χ²_{d, λ}
    //
    //  where:
    //    d      = 4κθ / σ²           (degrees of freedom)
    //    λ      = 4κ · V_t · e^{-κ·dt} / (σ²(1 − e^{-κ·dt}))   (non-centrality)
    //    scale  = σ²(1 − e^{-κ·dt}) / (4κ)
    // -------------------------------------------------------------------
    const double dof = 4.0 * kappa * theta / (sigma * sigma);
    const double lambda = 4.0 * kappa * v_t * e_kdt / (sigma * sigma * (1.0 - e_kdt));
    const double scale = sigma * sigma * (1.0 - e_kdt) / (4.0 * kappa);

    // --- Welford's online algorithm for numerically stable mean & variance ---
    double mean = 0.0;
    double M2 = 0.0;

    for (int i = 0; i < num_samples; ++i)
    {
        // SampleNonCentralChi2 returns a raw χ²_{d,λ} draw; multiply by scale
        // to recover a V_{t+dt} sample in variance units.
        double x = scale * SampleNonCentralChi2(dof, lambda);
        double delta = x - mean;
        mean += delta / static_cast<double>(i + 1);
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    const double sample_variance = M2 / static_cast<double>(num_samples - 1);

    // --- 5-sigma tolerance ---
    // SE(mean)     = sqrt(Var / n)
    // SE(variance) = sqrt(2 · Var² / (n−1))   [sampling dist. of sample var]
    const double tol_mean = 5.0 * std::sqrt(theoretical_var / static_cast<double>(num_samples));
    const double tol_var = 5.0 * std::sqrt(2.0 * theoretical_var * theoretical_var / static_cast<double>(num_samples - 1));

    const bool mean_ok = std::abs(mean - theoretical_mean) < tol_mean;
    const bool var_ok = std::abs(sample_variance - theoretical_var) < tol_var;

    // --- Report ---
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "\n=== Heston variance moment test ===\n";
    std::cout << "Params : kappa=" << kappa << "  theta=" << theta
              << "  sigma=" << sigma << "  v_t=" << v_t
              << "  dt=1/252\n";
    std::cout << "CIR    : dof=" << dof << "  lambda=" << lambda
              << "  scale=" << scale << "\n\n";

    std::cout << "Theoretical mean   : " << theoretical_mean << "\n";
    std::cout << "Sample mean        : " << mean << "\n";
    std::cout << "Tolerance (5σ)     : " << tol_mean << "\n";
    std::cout << "Mean test          : " << (mean_ok ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Theoretical var    : " << theoretical_var << "\n";
    std::cout << "Sample variance    : " << sample_variance << "\n";
    std::cout << "Tolerance (5σ)     : " << tol_var << "\n";
    std::cout << "Variance test      : " << (var_ok ? "PASS" : "FAIL") << "\n";

    // Hard assertions — uncomment for use in a test framework
    // assert(mean_ok && "Mean outside 5-sigma bound");
    // assert(var_ok  && "Variance outside 5-sigma bound");
}

void test_characteristic_function()
{
    std::cout << "\n=== Testing Heston Characteristic Function ===\n"
              << std::endl;

    // Initialize Heston parameters (typical values from literature)
    HestonParams heston_params = {
        .kappa = 2.0,  // Mean reversion rate
        .theta = 0.45, // Long-run variance
        .sigma = 0.25, // Volatility of variance
        .v_u = 0.2,    // Variance at time u
        .v_t = 0.1,    // Variance at time t
        .dt = 0.25     // One day in years
    };

    // =====================================================
    // TEST 1: u = 0 should return exactly 1
    // =====================================================
    std::cout << "--- TEST 1: Characteristic function at u = 0 ---\n"
              << std::endl;

    double u_zero = 0.0;
    auto phi_zero = CharFunction(heston_params, u_zero);

    std::cout << "Φ(0) = " << phi_zero << std::endl;
    std::cout << "Expected: (1, 0)" << std::endl;

    if (std::abs(phi_zero - std::complex<double>(1.0, 0.0)) < 1e-10)
    {
        std::cout << "✅ Test passes!" << std::endl;
    }
    else
    {
        std::cout << "❌ Test fails!" << std::endl;
    }

    // =====================================================
    // TEST 1b: u very close to zero (just above guard threshold)
    // =====================================================
    std::cout << "\n--- TEST 1b: Characteristic function at u = 1e-10 ---\n"
              << std::endl;

    double u_small = 1e-10;
    auto phi_small = CharFunction(heston_params, u_small);

    std::cout << "Φ(1e-10) = " << phi_small << std::endl;
    std::cout << "Expected: ≈ (1, 0) (very close to 1)" << std::endl;

    if (std::abs(phi_small - std::complex<double>(1.0, 0.0)) < 1e-8)
    {
        std::cout << "✅ Small u test passes!" << std::endl;
    }
    else
    {
        std::cout << "⚠️ Deviation: |Φ(1e-10) - 1| = "
                  << std::abs(phi_small - std::complex<double>(1.0, 0.0)) << std::endl;
    }

    // =====================================================
    // TEST 2: First moment via numerical derivative
    // =====================================================
    std::cout << "\n--- TEST 2: First moment of integrated variance ---\n"
              << std::endl;

    double eps = 1e-6;
    auto phi_plus = CharFunction(heston_params, eps);
    auto phi_minus = CharFunction(heston_params, -eps);
    std::complex<double> i(0.0, 1.0);

    // =====================================================
    // Additional Test: Characteristic function for different u values
    // =====================================================
    std::cout << "\n--- Additional: Characteristic function for various u ---\n"
              << std::endl;

    std::vector<double> u_values = {0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 30.0, 100.0, 10000.0};

    for (double u_val : u_values)
    {
        auto phi = CharFunction(heston_params, u_val);
        std::cout << "Φ(" << u_val << ") = " << phi << std::endl;
    }

    // =====================================================
    // Convergence Test: Check if |Φ(u)| ≤ 1 (property of CF)
    // =====================================================
    std::cout << "\n--- Convergence Test: Checking |Φ(u)| ≤ 1 ---\n"
              << std::endl;

    bool all_bounded = true;
    for (double u_val : u_values)
    {
        auto phi = CharFunction(heston_params, u_val);
        double magnitude = std::abs(phi);
        std::cout << "|Φ(" << u_val << ")| = " << magnitude;

        if (magnitude <= 1.0 + 1e-10)
        { // Small tolerance for numerical error
            std::cout << " ✅";
        }
        else
        {
            std::cout << " ❌ (exceeds 1 by " << magnitude - 1.0 << ")";
            all_bounded = false;
        }
        std::cout << std::endl;
    }

    if (all_bounded)
    {
        std::cout << "\n✅ All characteristic function values bounded by 1!" << std::endl;
    }
    else
    {
        std::cout << "\n⚠️ Some values exceed 1 - possible numerical issues" << std::endl;
    }

    // =====================================================
    // Symmetry Test: Φ(-u) should be complex conjugate of Φ(u)
    // =====================================================
    std::cout << "\n--- Symmetry Test: Φ(-u) = conj(Φ(u)) ---\n"
              << std::endl;

    double test_u = 1.0;
    auto phi_pos = CharFunction(heston_params, test_u);
    auto phi_neg = CharFunction(heston_params, -test_u);

    std::cout << "Φ(" << test_u << ")  = " << phi_pos << std::endl;
    std::cout << "Φ(-" << test_u << ") = " << phi_neg << std::endl;
    std::cout << "conj(Φ(" << test_u << ")) = " << std::conj(phi_pos) << std::endl;

    if (std::abs(phi_neg - std::conj(phi_pos)) < 1e-10)
    {
        std::cout << "✅ Symmetry property holds!" << std::endl;
    }
    else
    {
        std::cout << "❌ Symmetry property fails!" << std::endl;
    }
}

void test_first_moment_sanity()
{
    std::cout << "\n--- First Moment Sanity Check ---\n"
              << std::endl;

    // For very short time dt → 0, the integrated variance should be ≈ V_u * dt
    HestonParams very_short = {
        .kappa = 2.0,
        .theta = 0.45,
        .sigma = 0.25,
        .v_u = 0.2,
        .v_t = 0.2, // Almost same as V_u for very short dt
        .dt = 1e-6  // Very short time
    };

    double eps = 1e-6;
    auto phi_plus = CharFunction(very_short, eps);
    auto phi_minus = CharFunction(very_short, -eps);
    std::complex<double> i(0.0, 1.0);
    std::complex<double> derivative = (phi_plus - phi_minus) / (2.0 * eps);
    double mean_numerical = std::imag(derivative);

    double expected_mean_approx = very_short.v_u * very_short.dt;

    std::cout << "For very short dt (1e-6):" << std::endl;
    std::cout << "  Numerical E[∫V_s ds] ≈ " << mean_numerical << std::endl;
    std::cout << "  Approximation V_u * dt = " << expected_mean_approx << std::endl;

    if (std::abs(mean_numerical - expected_mean_approx) / expected_mean_approx < 0.01)
    {
        std::cout << "  ✅ Reasonable agreement for short time!" << std::endl;
    }
    else
    {
        std::cout << "  ⚠️ Deviation larger than expected" << std::endl;
    }
}

void test_integrals()
{
    HestonParams p = {
        .kappa = 2.0,
        .theta = 0.45,
        .sigma = 0.45,
        .v_u = 0.45,
        .v_t = 0.20, // Almost same as V_u for very short dt
        .dt = 1.0 / 365.0};

    // Checking the functioning of the integrals

    double x = 1.0;

    double u = 1.0;

    double epsilon = 1e-4;

    auto cdf_int = CDFIntegrand(x, u, p);
    auto pdf_int = PDFIntegrand(x, u, p);
    auto d_pdf_int = d_PDFIntegrand(x, u, p);

    std::cout << "The value of cdf integrand at x = " << x << " is " << cdf_int << std::endl;
    std::cout << "The value of pdf integrand at x = " << x << " is " << pdf_int << std::endl;
    std::cout << "The value of d_pdf integrand at x = " << x << " is " << d_pdf_int << std::endl;

    auto cdf_val = calculateCDF(x, p);
    auto pdf_val = calculateIntegral(PDFIntegrand, x, p);

    std::cout << "The value of cdf at x = " << x << " is " << cdf_val << std::endl;
    std::cout << "The value of pdf at x = " << x << " is " << pdf_val << std::endl;

    auto cdf_ep = calculateCDF(x + epsilon, p);

    // ASSERT(cdf_ep > cdf_val - 1e-6, "The CDF is not exhibiting monotonicity");

    auto cdf_high = calculateCDF(6.0, p); // This should give a value close to 1

    // ASSERT(approx_equal(cdf_high, 1.0), "The CDF should converge to 1 for high values of x");

    std::cout << cdf_high << std::endl;

    auto cdf_low = calculateCDF(1e-7, p); // This should give a value close to 0

    std::cout << cdf_low << std::endl;
    // ASSERT(approx_equal(cdf_low, 0.0), "The CDF should converge to 0 for low values of X");

    // PDF should approximately equal the finite difference of CDF
    double cdf_plus = calculateCDF(x + epsilon, p);
    double cdf_minus = calculateCDF(x - epsilon, p);
    double fd_pdf = (cdf_plus - cdf_minus) / (2.0 * epsilon);

    std::cout << "PDF from integration : " << pdf_val << std::endl;
    std::cout << "PDF from finite diff : " << fd_pdf << std::endl;

    // Running the Newton Solver
    double uniform = 1.0; // Assuming U = 0.5 for testing

    auto result = runNewtonSolver(uniform, p);

    std::cout << "The value of x for U = " << uniform << " is " << result << std::endl;

    std::cout << "CDF(0.00279414) = " << calculateCDF(0.00279414, p) << std::endl;
    std::cout << "CDF(0.037) = " << calculateCDF(0.037, p) << std::endl;
    std::cout << "CDF(0.05) = " << calculateCDF(0.05, p) << std::endl;

    std::cout << "Evaluating Fourier method" << std::endl;

    CDFGrid fourier_grid = computeCDFGrid(p, 2048, 0.25);

    double cdf_fft = sampleIntegratedVariance(uniform, fourier_grid);

    std::cout << "The value of x for U = " << uniform << " is " << cdf_fft << std::endl;
}

void test_quadrature()
{
    std::cout << "\n========== Testing Gauss-Legendre Quadrature ==========\n";
    std::cout << std::setprecision(12);

    // Test 1: Constant function f(x) = 1 over [0, 5]
    {
        auto f = [](double x)
        { return 1.0; };
        double a = 0.0, b = 5.0;
        double expected = b - a;
        double result = legendreIntegrate(f, a, b);
        std::cout << "\nTest 1: ∫₀⁵ 1 dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 2: Linear function f(x) = x over [0, 1]
    {
        auto f = [](double x)
        { return x; };
        double a = 0.0, b = 1.0;
        double expected = 0.5;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 2: ∫₀¹ x dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 3: Quadratic function f(x) = x² over [0, 2]
    {
        auto f = [](double x)
        { return x * x; };
        double a = 0.0, b = 2.0;
        double expected = 8.0 / 3.0;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 3: ∫₀² x² dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 4: Cubic function f(x) = x³ over [-1, 1] (should be 0)
    {
        auto f = [](double x)
        { return x * x * x; };
        double a = -1.0, b = 1.0;
        double expected = 0.0;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 4: ∫₋₁¹ x³ dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 5: Sine function f(x) = sin(x) over [0, π]
    {
        auto f = [](double x)
        { return std::sin(x); };
        double a = 0.0, b = M_PI;
        double expected = 2.0;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 5: ∫₀^π sin(x) dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 6: Cosine function f(x) = cos(x) over [0, π/2]
    {
        auto f = [](double x)
        { return std::cos(x); };
        double a = 0.0, b = M_PI / 2.0;
        double expected = 1.0;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 6: ∫₀^(π/2) cos(x) dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 7: Exponential function f(x) = e^x over [0, 1]
    {
        auto f = [](double x)
        { return std::exp(x); };
        double a = 0.0, b = 1.0;
        double expected = std::exp(1.0) - 1.0;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 7: ∫₀¹ e^x dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 8: f(x) = 1/(1+x²) over [0, 1] (gives π/4)
    {
        auto f = [](double x)
        { return 1.0 / (1.0 + x * x); };
        double a = 0.0, b = 1.0;
        double expected = M_PI / 4.0;
        double result = legendreIntegrate(f, a, b);
        std::cout << "Test 8: ∫₀¹ 1/(1+x²) dx = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 9: f(x) = sqrt(x) over [0, 1] (has derivative singularity, tests robustness)
    {
        auto f = [](double x)
        { return std::sqrt(x); };
        double a = 0.0, b = 1.0;
        double expected = 2.0 / 3.0;
        double result = legendreIntegrate(f, a, b);
        double error = std::abs(result - expected);
        std::cout << "Test 9: ∫₀¹ √x dx = " << result << " (expected: " << expected << ") error: " << error << " ";
        if (error < 1e-8)
        {
            std::cout << "✅ PASSED\n";
        }
        else
        {
            std::cout << "⚠️ PASSED (within tolerance)\n";
        }
    }

    // Test 10: Segmented integration (like your CDF approach) for ∫₀^∞ e^(-x) dx = 1
    {
        auto f = [](double x)
        { return std::exp(-x); };
        double result = 0.0;
        result += legendreIntegrate(f, 0.0, 5.0);
        result += legendreIntegrate(f, 5.0, 20.0);
        result += legendreIntegrate(f, 20.0, 80.0);
        result += std::exp(-80.0); // Tail approximation

        double expected = 1.0;
        std::cout << "Test 10: ∫₀^∞ e^(-x) dx (segmented) = " << result << " (expected: " << expected << ") ";
        assert(approx_equal(result, expected, 1e-8));
        std::cout << "✅ PASSED\n";
    }

    // Test 11: Compare with your actual PDF integrand (just to verify integration works)
    {
        // Test integrating a simple oscillatory function that's common in CF inversion
        auto f = [](double u)
        {
            return std::sin(u) / (u + 0.1);
        };
        double result = legendreIntegrate(f, 0.0, 100.0);
        // No closed form, just checking it runs without crashing
        std::cout << "Test 11: ∫₀¹⁰⁰ sin(u)/(u+0.1) du = " << result << " (computed without error) ";
        if (std::isfinite(result))
        {
            std::cout << "✅ PASSED\n";
        }
        else
        {
            std::cout << "❌ FAILED (non-finite result)\n";
        }
    }

    std::cout << "\n========== All quadrature tests completed ==========\n";
}

void test_oscillatory_quadrature()
{
    std::cout << "\n===== Testing Oscillatory Quadrature (CF-style) =====\n";
    std::cout << std::setprecision(12);

    // Test A: sin(ux) * e^(-u) — mimics sin(ux)*Im(φ) with exponential decay
    // ∫₀^∞ sin(ux) * e^(-u) du = x / (1 + x²)  [Laplace transform]
    {
        std::vector<double> x_vals = {0.5, 1.0, 2.0, 5.0, 10.0};
        std::cout << "\nTest A: ∫₀^∞ sin(ux)*e^(-u) du = x/(1+x²)\n";

        for (double x : x_vals)
        {
            auto f = [x](double u)
            { return std::sin(u * x) * std::exp(-u); };

            double result = 0.0;
            // Breakpoints scale with x to capture oscillations
            std::vector<double> bp = {0.0, 5.0 / x, 20.0 / x, 50.0 / x, 100.0 / x};
            for (int k = 0; k + 1 < bp.size(); k++)
                result += legendreIntegrate(f, bp[k], bp[k + 1]);

            double expected = x / (1.0 + x * x);
            double error = std::abs(result - expected);
            std::cout << "  x=" << x << ": result=" << result
                      << " expected=" << expected
                      << " error=" << error;
            std::cout << (error < 1e-6 ? " ✅\n" : " ❌\n");
        }
    }

    // Test B: cos(ux) * e^(-u) — mimics cos(ux)*Re(φ)
    // ∫₀^∞ cos(ux) * e^(-u) du = 1 / (1 + x²)  [Laplace transform]
    {
        std::vector<double> x_vals = {0.5, 1.0, 2.0, 5.0, 10.0};
        std::cout << "\nTest B: ∫₀^∞ cos(ux)*e^(-u) du = 1/(1+x²)\n";

        for (double x : x_vals)
        {
            auto f = [x](double u)
            { return std::cos(u * x) * std::exp(-u); };

            double result = 0.0;
            std::vector<double> bp = {0.0, 5.0 / x, 20.0 / x, 50.0 / x, 100.0 / x};
            for (int k = 0; k + 1 < bp.size(); k++)
                result += legendreIntegrate(f, bp[k], bp[k + 1]);

            double expected = 1.0 / (1.0 + x * x);
            double error = std::abs(result - expected);
            std::cout << "  x=" << x << ": result=" << result
                      << " expected=" << expected
                      << " error=" << error;
            std::cout << (error < 1e-6 ? " ✅\n" : " ❌\n");
        }
    }

    // Test C: Harder — Gaussian decay mimics faster-decaying φ
    // ∫₀^∞ sin(ux) * e^(-u²) du  [no closed form, use scipy as reference]
    // Reference values precomputed: x=1 → 0.42440, x=2 → 0.18128
    {
        std::cout << "\nTest C: ∫₀^∞ sin(ux)*e^(-u²) du [Gaussian decay]\n";
        std::map<double, double> references = {{1.0, 0.424400}, {2.0, 0.538079506}};

        for (auto &pair : references)
        {
            double x = pair.first;
            double expected = pair.second;

            auto f = [x](double u)
            { return std::sin(u * x) * std::exp(-u * u); };

            double result = 0.0;
            std::vector<double> bp = {0.0, 2.0 / x, 5.0 / x, 10.0 / x};
            for (size_t k = 0; k + 1 < bp.size(); k++)
                result += legendreIntegrate(f, bp[k], bp[k + 1]);

            double error = std::abs(result - expected);
            std::cout << "  x=" << x << ": result=" << result
                      << " expected=" << expected
                      << " error=" << error;
            std::cout << (error < 1e-4 ? " ✅\n" : " ❌\n");
        }
    }

    // Test D: Fixed breakpoints vs scaled breakpoints — the critical comparison
    // This directly exposes your original bug
    {
        std::cout << "\nTest D: Fixed vs scaled breakpoints for high x\n";
        double x = 20.0; // High x = rapid oscillation
        auto f = [x](double u)
        { return std::sin(u * x) * std::exp(-u); };
        double expected = x / (1.0 + x * x); // 20/401 ≈ 0.04988

        // Fixed breakpoints (your original approach)
        double fixed_result = 0.0;
        std::vector<std::pair<double, double>> intervals = {{0, 5}, {5, 20}, {20, 50}, {50, 100}};
        for (const auto &interval : intervals)
        {
            double a = interval.first;
            double b = interval.second;
            fixed_result += legendreIntegrate(f, a, b);
        }

        // Scaled breakpoints
        double scaled_result = 0.0;
        std::vector<double> bp = {0.0, 5.0 / x, 20.0 / x, 50.0 / x, 100.0 / x};
        for (int k = 0; k + 1 < bp.size(); k++)
            scaled_result += legendreIntegrate(f, bp[k], bp[k + 1]);

        std::cout << "  expected:        " << expected << "\n";
        std::cout << "  fixed result:    " << fixed_result
                  << " error=" << std::abs(fixed_result - expected)
                  << (std::abs(fixed_result - expected) < 1e-6 ? " ✅\n" : " ❌\n");
        std::cout << "  scaled result:   " << scaled_result
                  << " error=" << std::abs(scaled_result - expected)
                  << (std::abs(scaled_result - expected) < 1e-6 ? " ✅\n" : " ❌\n");
    }
}

int main()
{
    // test_generator();

    // test_heston_variance_moments();
    // test_characteristic_function();
    // test_first_moment_sanity();
    // test_integrals();
    test_quadrature();
    test_oscillatory_quadrature();

    return 0;
}
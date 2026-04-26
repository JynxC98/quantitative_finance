/**
 * @brief This script is used to test the implementations
 */

#include <iostream>
#include <cmath>
#include <iomanip>

#include "bessel.hpp"
#include "non_central_chi_sqd.hpp"

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
int main()
{
    test_generator();
    test_heston_variance_moments();
    return 0;
}
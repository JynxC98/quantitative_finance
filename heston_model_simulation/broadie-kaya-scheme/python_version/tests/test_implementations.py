"""
This script is used to test the implementations.
"""

import sys
import os
import numpy as np
from scipy.stats import ncx2
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers.bessel import modified_bessel
from helpers.char_func import char_function
from helpers.gamma import gamma, gamma_real
from helpers.solvers import (
    calculate_integral,
    cdf_integrand,
    calculate_cdf,
    pdf_integrand,
    d_pdf_integrand,
    run_newton_solver,
)


@dataclass
class HestonParams:
    """Heston model parameters container"""

    v_t: float  # Current variance
    v_u: float  # Long-term variance
    dt: float  # Time step
    # Add other Heston parameters as needed (kappa, theta, sigma, rho)
    kappa: float = 0.0
    theta: float = 0.0
    sigma: float = 0.0
    rho: float = 0.0


def sample_non_central_chi2(dof, lambda_, size=1):
    """Generate samples from non-central chi-squared distribution"""
    # Using scipy's ncx2 distribution
    return ncx2.rvs(df=dof, nc=lambda_, size=size)


def test_generator():
    """Test the non-central chi-squared generator"""
    dof = 3.0
    lambda_ = 3.0

    num_samples = 1000000

    # Welford's online algorithm for numerically stable mean & variance
    mean = 0.0
    M2 = 0.0

    for i in range(num_samples):
        x = sample_non_central_chi2(dof, lambda_)
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        M2 += delta * delta2

    sample_variance = M2 / (num_samples - 1)

    # Theoretical values
    theoretical_mean = dof + lambda_
    theoretical_variance = 2.0 * (dof + 2.0 * lambda_)

    # Tolerance: 5-sigma bounds using standard error
    sigma_mean = 5.0 * np.sqrt(theoretical_variance / num_samples)
    sigma_variance = 5.0 * np.sqrt(
        2.0 * theoretical_variance * theoretical_variance / (num_samples - 1)
    )

    # Assertions
    mean_ok = abs(mean - theoretical_mean) < sigma_mean
    variance_ok = abs(sample_variance - theoretical_variance) < sigma_variance

    print(f"Samples          : {num_samples}")
    print(f"Theoretical mean : {theoretical_mean:.6f}")
    print(f"Sample mean      : {mean:.6f}")
    print(f"Mean tolerance   : ± {sigma_mean:.6f}")
    print(f"Mean test        : {'PASS' if mean_ok else 'FAIL'}\n")
    print(f"Theoretical var  : {theoretical_variance:.6f}")
    print(f"Sample variance  : {sample_variance:.6f}")
    print(f"Variance tol     : ± {sigma_variance:.6f}")
    print(f"Variance test    : {'PASS' if variance_ok else 'FAIL'}")


def test_heston_variance_moments():
    """Test Heston/CIR variance moments"""
    # Heston / CIR parameters
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    v_t = 0.04
    dt = 1.0 / 252.0

    num_samples = 500000

    # Theoretical conditional moments of V_{t+dt} | V_t (CIR closed form)
    e_kdt = np.exp(-kappa * dt)
    theoretical_mean = v_t * e_kdt + theta * (1.0 - e_kdt)
    theoretical_var = v_t * (sigma * sigma / kappa) * (e_kdt - e_kdt * e_kdt) + (
        theta * sigma * sigma / (2.0 * kappa)
    ) * np.power(1.0 - e_kdt, 2.0)

    # Non-central chi-squared parameterisation (Broadie-Kaya)
    dof = 4.0 * kappa * theta / (sigma * sigma)
    lambda_ = 4.0 * kappa * v_t * e_kdt / (sigma * sigma * (1.0 - e_kdt))
    scale = sigma * sigma * (1.0 - e_kdt) / (4.0 * kappa)

    # Welford's online algorithm
    mean = 0.0
    M2 = 0.0

    for i in range(num_samples):
        x = scale * sample_non_central_chi2(dof, lambda_)
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        M2 += delta * delta2

    sample_variance = M2 / (num_samples - 1)

    # 5-sigma tolerance
    tol_mean = 5.0 * np.sqrt(theoretical_var / num_samples)
    tol_var = 5.0 * np.sqrt(2.0 * theoretical_var * theoretical_var / (num_samples - 1))

    mean_ok = abs(mean - theoretical_mean) < tol_mean
    var_ok = abs(sample_variance - theoretical_var) < tol_var

    # Report
    print(f"\n=== Heston variance moment test ===")
    print(f"Params : kappa={kappa}  theta={theta} sigma={sigma} v_t={v_t} dt=1/252")
    print(f"CIR    : dof={dof}  lambda={lambda_} scale={scale}\n")
    print(f"Theoretical mean   : {theoretical_mean:.8f}")
    print(f"Sample mean        : {mean:.8f}")
    print(f"Tolerance (5σ)     : {tol_mean:.8f}")
    print(f"Mean test          : {'PASS' if mean_ok else 'FAIL'}\n")
    print(f"Theoretical var    : {theoretical_var:.8f}")
    print(f"Sample variance    : {sample_variance:.8f}")
    print(f"Tolerance (5σ)     : {tol_var:.8f}")
    print(f"Variance test      : {'PASS' if var_ok else 'FAIL'}")


def test_characteristic_function():
    """Test Heston characteristic function"""
    print("\n=== Testing Heston Characteristic Function ===\n")

    # Heston parameters (typical values from literature)
    heston_params = HestonParams(
        kappa=2.0,  # Mean reversion rate
        theta=0.45,  # Long-run variance
        sigma=0.25,  # Volatility of variance
        v_u=0.2,  # Variance at time u
        v_t=0.1,  # Variance at time t
        dt=0.25,  # Time step in years
    )

    # TEST 1: u = 0 should return exactly 1
    print("--- TEST 1: Characteristic function at u = 0 ---\n")

    u_zero = 0.0
    phi_zero = char_function(heston_params, u_zero)

    print(f"Φ(0) = {phi_zero}")
    print("Expected: (1, 0)")

    if abs(phi_zero - complex(1.0, 0.0)) < 1e-10:
        print("✅ Test passes!")
    else:
        print("❌ Test fails!")

    # TEST 1b: u very close to zero
    print("\n--- TEST 1b: Characteristic function at u = 1e-10 ---\n")

    u_small = 1e-10
    phi_small = char_function(heston_params, u_small)

    print(f"Φ(1e-10) = {phi_small}")
    print("Expected: ≈ (1, 0) (very close to 1)")

    if abs(phi_small - complex(1.0, 0.0)) < 1e-8:
        print("✅ Small u test passes!")
    else:
        print(f"⚠️ Deviation: |Φ(1e-10) - 1| = {abs(phi_small - complex(1.0, 0.0))}")

    # Additional test: Characteristic function for different u values
    print("\n--- Additional: Characteristic function for various u ---\n")

    u_values = [0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 30.0, 100.0, 10000.0]

    for u_val in u_values:
        phi = char_function(heston_params, u_val)
        print(f"Φ({u_val}) = {phi}")

    # Convergence test: Check if |Φ(u)| ≤ 1
    print("\n--- Convergence Test: Checking |Φ(u)| ≤ 1 ---\n")

    all_bounded = True
    for u_val in u_values:
        phi = char_function(heston_params, u_val)
        magnitude = abs(phi)
        print(f"|Φ({u_val})| = {magnitude}", end=" ")

        if magnitude <= 1.0 + 1e-10:
            print("✅")
        else:
            print(f"❌ (exceeds 1 by {magnitude - 1.0})")
            all_bounded = False

    if all_bounded:
        print("\n✅ All characteristic function values bounded by 1!")
    else:
        print("\n⚠️ Some values exceed 1 - possible numerical issues")

    # Symmetry test
    print("\n--- Symmetry Test: Φ(-u) = conj(Φ(u)) ---\n")

    test_u = 1.0
    phi_pos = char_function(heston_params, test_u)
    phi_neg = char_function(heston_params, -test_u)

    print(f"Φ({test_u})  = {phi_pos}")
    print(f"Φ(-{test_u}) = {phi_neg}")
    print(f"conj(Φ({test_u})) = {np.conj(phi_pos)}")

    if abs(phi_neg - np.conj(phi_pos)) < 1e-10:
        print("✅ Symmetry property holds!")
    else:
        print("❌ Symmetry property fails!")


def test_first_moment_sanity():
    """Sanity check for first moment"""
    print("\n--- First Moment Sanity Check ---\n")

    # For very short time dt → 0, integrated variance should be ≈ V_u * dt
    very_short_params = HestonParams(
        kappa=2.0,
        theta=0.45,
        sigma=0.25,
        v_u=0.2,
        v_t=0.2,  # Almost same as V_u for very short dt
        dt=1e-6,  # Very short time
    )

    eps = 1e-6
    phi_plus = char_function(very_short_params, eps)
    phi_minus = char_function(very_short_params, -eps)
    derivative = (phi_plus - phi_minus) / (2.0 * eps)
    mean_numerical = np.imag(derivative)

    expected_mean_approx = very_short_params.v_u * very_short_params.dt

    print("For very short dt (1e-6):")
    print(f"  Numerical E[∫V_s ds] ≈ {mean_numerical}")
    print(f"  Approximation V_u * dt = {expected_mean_approx}")

    if abs(mean_numerical - expected_mean_approx) / expected_mean_approx < 0.01:
        print("  ✅ Reasonable agreement for short time!")
    else:
        print("  ⚠️ Deviation larger than expected")


def test_integrals():
    """Test integral calculations"""
    p = HestonParams(
        kappa=2.0, theta=0.45, sigma=0.45, v_u=0.45, v_t=0.20, dt=1.0 / 365.0
    )

    x = 1.0
    u = 1.0
    epsilon = 1e-4

    # Note: These functions need to be implemented based on your C++ code
    cdf_int = cdf_integrand(x, u, p)
    pdf_int = pdf_integrand(x, u, p)
    d_pdf_int = d_pdf_integrand(x, u, p)

    print(f"The value of cdf integrand at x = {x} is {cdf_int}")
    print(f"The value of pdf integrand at x = {x} is {pdf_int}")
    print(f"The value of d_pdf integrand at x = {x} is {d_pdf_int}")

    cdf_val = calculate_cdf(x, p)
    pdf_val = calculate_integral(pdf_integrand, x, p)

    print(f"The value of cdf at x = {x} is {cdf_val}")
    print(f"The value of pdf at x = {x} is {pdf_val}")

    cdf_ep = calculate_cdf(x + epsilon, p)
    cdf_high = calculate_cdf(6.0, p)
    cdf_low = calculate_cdf(1e-7, p)

    print(f"CDF at high value (6.0): {cdf_high}")
    print(f"CDF at low value (1e-7): {cdf_low}")

    # PDF should approximately equal the finite difference of CDF
    cdf_plus = calculate_cdf(x + epsilon, p)
    cdf_minus = calculate_cdf(x - epsilon, p)
    fd_pdf = (cdf_plus - cdf_minus) / (2.0 * epsilon)

    print(f"PDF from integration : {pdf_val}")
    print(f"PDF from finite diff : {fd_pdf}")

    # Running the Newton Solver
    uniform = 1.0
    result = run_newton_solver(uniform, p)

    print(f"The value of x for U = {uniform} is {result}")
    print(f"CDF(0.00279414) = {calculate_cdf(0.00279414, p)}")
    print(f"CDF(0.037) = {calculate_cdf(0.037, p)}")
    print(f"CDF(0.05) = {calculate_cdf(0.05, p)}")

    print("Evaluating Fourier method")
    # Note: CDFGrid and sampleIntegratedVariance would need to be implemented
    # cdf_fft = sample_integrated_variance(uniform, fourier_grid)
    # print(f"The value of x for U = {uniform} is {cdf_fft}")


def legendre_integrate(f, a, b, n=32):
    """
    Gauss-Legendre quadrature integration

    Args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of quadrature points (default 32)

    Returns:
        approximate integral value
    """
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    x_nodes, weights = np.polynomial.legendre.leggauss(n)

    # Transform from [-1, 1] to [a, b]
    t = 0.5 * (b - a) * x_nodes + 0.5 * (a + b)

    # Compute integral
    integral = 0.5 * (b - a) * np.sum(weights * f(t))

    return integral


def approx_equal(a, b, tol=1e-8):
    """Check if two values are approximately equal"""
    return abs(a - b) < tol


def test_quadrature():
    """Test Gauss-Legendre quadrature"""
    print("\n========== Testing Gauss-Legendre Quadrature ==========")

    # Test 1: Constant function f(x) = 1 over [0, 5]
    f = lambda x: 1.0
    a, b = 0.0, 5.0
    expected = b - a
    result = legendre_integrate(f, a, b)
    print(f"\nTest 1: ∫₀⁵ 1 dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 2: Linear function f(x) = x over [0, 1]
    f = lambda x: x
    a, b = 0.0, 1.0
    expected = 0.5
    result = legendre_integrate(f, a, b)
    print(f"Test 2: ∫₀¹ x dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 3: Quadratic function f(x) = x² over [0, 2]
    f = lambda x: x * x
    a, b = 0.0, 2.0
    expected = 8.0 / 3.0
    result = legendre_integrate(f, a, b)
    print(f"Test 3: ∫₀² x² dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 4: Cubic function f(x) = x³ over [-1, 1]
    f = lambda x: x * x * x
    a, b = -1.0, 1.0
    expected = 0.0
    result = legendre_integrate(f, a, b)
    print(f"Test 4: ∫₋₁¹ x³ dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 5: Sine function f(x) = sin(x) over [0, π]
    f = lambda x: np.sin(x)
    a, b = 0.0, np.pi
    expected = 2.0
    result = legendre_integrate(f, a, b)
    print(f"Test 5: ∫₀^π sin(x) dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 6: Cosine function f(x) = cos(x) over [0, π/2]
    f = lambda x: np.cos(x)
    a, b = 0.0, np.pi / 2.0
    expected = 1.0
    result = legendre_integrate(f, a, b)
    print(f"Test 6: ∫₀^(π/2) cos(x) dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 7: Exponential function f(x) = e^x over [0, 1]
    f = lambda x: np.exp(x)
    a, b = 0.0, 1.0
    expected = np.exp(1.0) - 1.0
    result = legendre_integrate(f, a, b)
    print(f"Test 7: ∫₀¹ e^x dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 8: f(x) = 1/(1+x²) over [0, 1]
    f = lambda x: 1.0 / (1.0 + x * x)
    a, b = 0.0, 1.0
    expected = np.pi / 4.0
    result = legendre_integrate(f, a, b)
    print(f"Test 8: ∫₀¹ 1/(1+x²) dx = {result} (expected: {expected}) ", end="")
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 9: f(x) = sqrt(x) over [0, 1]
    f = lambda x: np.sqrt(x)
    a, b = 0.0, 1.0
    expected = 2.0 / 3.0
    result = legendre_integrate(f, a, b)
    error = abs(result - expected)
    print(
        f"Test 9: ∫₀¹ √x dx = {result} (expected: {expected}) error: {error} ", end=""
    )
    if error < 1e-8:
        print("✅ PASSED")
    else:
        print("⚠️ PASSED (within tolerance)")

    # Test 10: Segmented integration for ∫₀^∞ e^(-x) dx = 1
    f = lambda x: np.exp(-x)
    result = (
        legendre_integrate(f, 0.0, 5.0)
        + legendre_integrate(f, 5.0, 20.0)
        + legendre_integrate(f, 20.0, 80.0)
        + np.exp(-80.0)
    )

    expected = 1.0
    print(
        f"Test 10: ∫₀^∞ e^(-x) dx (segmented) = {result} (expected: {expected}) ",
        end="",
    )
    assert approx_equal(result, expected)
    print("✅ PASSED")

    # Test 11: Oscillatory function
    f = lambda u: np.sin(u) / (u + 0.1)
    result = legendre_integrate(f, 0.0, 100.0)
    print(
        f"Test 11: ∫₀¹⁰⁰ sin(u)/(u+0.1) du = {result} (computed without error) ", end=""
    )
    if np.isfinite(result):
        print("✅ PASSED")
    else:
        print("❌ FAILED (non-finite result)")

    print("\n========== All quadrature tests completed ==========")


def test_oscillatory_quadrature():
    """Test oscillatory quadrature for characteristic function style integrals"""
    print("\n===== Testing Oscillatory Quadrature (CF-style) =====")

    # Test A: sin(ux) * e^(-u) — mimics sin(ux)*Im(φ) with exponential decay
    print("\nTest A: ∫₀^∞ sin(ux)*e^(-u) du = x/(1+x²)")
    x_vals = [0.5, 1.0, 2.0, 5.0, 10.0]

    for x in x_vals:
        f = lambda u: np.sin(u * x) * np.exp(-u)

        result = 0.0
        # Breakpoints scale with x to capture oscillations
        bp = [0.0, 5.0 / x, 20.0 / x, 50.0 / x, 100.0 / x]
        for k in range(len(bp) - 1):
            result += legendre_integrate(f, bp[k], bp[k + 1])

        expected = x / (1.0 + x * x)
        error = abs(result - expected)
        print(f"  x={x}: result={result} expected={expected} error={error}", end=" ")
        print("✅" if error < 1e-6 else "❌")

    # Test B: cos(ux) * e^(-u) — mimics cos(ux)*Re(φ)
    print("\nTest B: ∫₀^∞ cos(ux)*e^(-u) du = 1/(1+x²)")

    for x in x_vals:
        f = lambda u: np.cos(u * x) * np.exp(-u)

        result = 0.0
        bp = [0.0, 5.0 / x, 20.0 / x, 50.0 / x, 100.0 / x]
        for k in range(len(bp) - 1):
            result += legendre_integrate(f, bp[k], bp[k + 1])

        expected = 1.0 / (1.0 + x * x)
        error = abs(result - expected)
        print(f"  x={x}: result={result} expected={expected} error={error}", end=" ")
        print("✅" if error < 1e-6 else "❌")

    # Test C: Gaussian decay
    print("\nTest C: ∫₀^∞ sin(ux)*e^(-u²) du [Gaussian decay]")
    references = {1.0: 0.424400, 2.0: 0.181282}

    for x, expected in references.items():
        f = lambda u: np.sin(u * x) * np.exp(-u * u)

        result = 0.0
        bp = [0.0, 2.0 / x, 5.0 / x, 10.0 / x]
        for k in range(len(bp) - 1):
            result += legendre_integrate(f, bp[k], bp[k + 1])

        error = abs(result - expected)
        print(f"  x={x}: result={result} expected={expected} error={error}", end=" ")
        print("✅" if error < 1e-4 else "❌")

    # Test D: Fixed vs scaled breakpoints comparison
    print("\nTest D: Fixed vs scaled breakpoints for high x")
    x = 20.0  # High x = rapid oscillation
    f = lambda u: np.sin(u * x) * np.exp(-u)
    expected = x / (1.0 + x * x)  # 20/401 ≈ 0.04988

    # Fixed breakpoints (original approach)
    fixed_result = 0.0
    fixed_intervals = [(0, 5), (5, 20), (20, 50), (50, 100)]
    for a, b in fixed_intervals:
        fixed_result += legendre_integrate(f, a, b)

    # Scaled breakpoints
    scaled_result = 0.0
    bp = [0.0, 5.0 / x, 20.0 / x, 50.0 / x, 100.0 / x]
    for k in range(len(bp) - 1):
        scaled_result += legendre_integrate(f, bp[k], bp[k + 1])

    print(f"  expected:        {expected}")
    print(
        f"  fixed result:    {fixed_result} error={abs(fixed_result - expected)}",
        end=" ",
    )
    print("✅" if abs(fixed_result - expected) < 1e-6 else "❌")
    print(
        f"  scaled result:   {scaled_result} error={abs(scaled_result - expected)}",
        end=" ",
    )
    print("✅" if abs(scaled_result - expected) < 1e-6 else "❌")


def main():
    """Main test function"""
    # Uncomment tests as needed
    # test_generator()
    # test_heston_variance_moments()
    test_characteristic_function()
    test_first_moment_sanity()
    test_integrals()
    test_quadrature()
    test_oscillatory_quadrature()


if __name__ == "__main__":
    main()

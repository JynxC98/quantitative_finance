"""
test_solvers.py
---------------
Mirrors the C++ test_integrals() test suite, ported to Python using the
same check/section helpers as the rest of the test suite.

Covers:
  1. Integrand spot-values  (cdf, pdf, d_pdf at a single (x, u) point)
  2. CDF monotonicity       (F(x+ε) > F(x))
  3. CDF boundary behaviour (F → 0 as x → 0⁺,  F → 1 as x → ∞)
  4. CDF grid sweep         (same x-values as the C++ loop)
  5. CDF at the mean        (should sit near 0.5 for symmetric params)
  6. PDF–CDF consistency    (PDF ≈ finite-difference of CDF)
  7. Halley solver round-trip (F(Q(u)) ≈ u for several quantiles)
  8. Brent solver round-trip  (same, as a reference cross-check)
"""

import numpy as np
from char_func import HestonParams
from solvers import (
    cdf_integrand,
    pdf_integrand,
    d_pdf_integrand,
    _calculate_integral,
    calculate_cdf,
    calculate_pdf,
    run_halley_solver,
    run_brent_solver,
)

# ---------------------------------------------------------------------------
# Shared fixtures  (match the C++ test exactly)
# ---------------------------------------------------------------------------

P = HestonParams(
    kappa=2.0,
    theta=0.20,
    sigma=0.45,
    v_t=0.20,  # V_0  (starting variance)
    v_u=0.20,  # V_dt (ending variance, same → stationary point)
    dt=0.25,  # quarterly step
)

X = 0.25  # primary evaluation point
U = 1.0  # frequency for integrand spot-checks
EPSILON = 1e-4  # finite-difference step
MEAN_VAR = 0.5 * (P.v_t + P.v_u) * P.dt  # trapezoidal mean ≈ 0.025

# x-grid from the C++ loop
X_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# Helpers  (kept identical to the rest of the test suite)
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}]  {label}{suffix}")
    if not condition:
        # Accumulate failures so the whole suite still runs
        check._failures.append(label)


check._failures = []


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------


def test_solvers():

    # ------------------------------------------------------------------
    # 1. Integrand spot-values
    # ------------------------------------------------------------------
    section("1. Integrand spot-values at (x=0.25, u=1.0)")

    cdf_int = cdf_integrand(X, U, P)
    pdf_int = pdf_integrand(X, U, P)
    d_pdf_int = d_pdf_integrand(X, U, P)

    print(f"  cdf_integrand  = {cdf_int:.8f}")
    print(f"  pdf_integrand  = {pdf_int:.8f}")
    print(f"  d_pdf_integrand= {d_pdf_int:.8f}")

    check("cdf_integrand returns finite float", np.isfinite(cdf_int), f"got {cdf_int}")
    check("pdf_integrand returns finite float", np.isfinite(pdf_int), f"got {pdf_int}")
    check(
        "d_pdf_integrand returns finite float",
        np.isfinite(d_pdf_int),
        f"got {d_pdf_int}",
    )
    check("pdf_integrand > 0 (positive density)", pdf_int > 0, f"got {pdf_int}")

    # ------------------------------------------------------------------
    # 2. CDF / PDF full integrals at X
    # ------------------------------------------------------------------
    section("2. CDF and PDF at x=0.25")

    cdf_val = calculate_cdf(X, P)
    pdf_val = calculate_pdf(X, P)

    print(f"  CDF({X}) = {cdf_val:.8f}")
    print(f"  PDF({X}) = {pdf_val:.8f}")

    check("CDF in (0, 1)", 0.0 < cdf_val < 1.0, f"got {cdf_val:.8f}")
    check("PDF > 0", pdf_val > 0, f"got {pdf_val:.8f}")

    # calculate_cdf must equal 0.5 + ∫ cdf_integrand
    raw_integral = _calculate_integral(cdf_integrand, X, P)
    check(
        "calculate_cdf == 0.5 + _calculate_integral(cdf_integrand)",
        abs((0.5 + raw_integral) - cdf_val) < 1e-10,
        f"0.5+integral={0.5+raw_integral:.10f}, direct={cdf_val:.10f}",
    )

    # ------------------------------------------------------------------
    # 3. CDF monotonicity
    # ------------------------------------------------------------------
    section("3. CDF monotonicity  F(x+ε) > F(x)")

    cdf_ep = calculate_cdf(X + EPSILON, P)
    check(
        f"CDF({X+EPSILON:.4f}) > CDF({X})",
        cdf_ep > cdf_val,
        f"{cdf_ep:.8f} vs {cdf_val:.8f}",
    )

    # ------------------------------------------------------------------
    # 4. CDF boundary behaviour
    # ------------------------------------------------------------------
    section("4. CDF boundary behaviour")

    cdf_high = calculate_cdf(6.0, P)
    cdf_low = calculate_cdf(1e-7, P)

    print(f"  CDF(6.0)  = {cdf_high:.8f}  (expect ≈ 1)")
    print(f"  CDF(1e-7) = {cdf_low:.8f}   (expect ≈ 0)")

    check("CDF(6.0)  > 0.99", cdf_high > 0.99, f"got {cdf_high:.8f}")
    check("CDF(1e-7) < 0.01", cdf_low < 0.01, f"got {cdf_low:.8f}")

    # ------------------------------------------------------------------
    # 5. CDF grid sweep  (mirrors the C++ for-loop)
    # ------------------------------------------------------------------
    section("5. CDF grid sweep — monotone increasing")

    prev = 0.0
    for x in X_GRID:
        val = calculate_cdf(x, P)
        print(f"  CDF({x:.2f}) = {val:.8f}")
        check(f"CDF({x}) > CDF(prev)", val > prev, f"{val:.8f} vs {prev:.8f}")
        prev = val

    # ------------------------------------------------------------------
    # 6. CDF at the trapezoidal mean
    # ------------------------------------------------------------------
    section("6. CDF at trapezoidal mean")

    cdf_mean = calculate_cdf(MEAN_VAR, P)
    print(f"  mean_var = {MEAN_VAR:.6f},  CDF(mean) = {cdf_mean:.8f}")

    # For v_t == v_u the distribution is not symmetric, but the mean
    # should sit somewhere in the interior — not near 0 or 1.
    check("CDF(mean) in (0.05, 0.95)", 0.05 < cdf_mean < 0.95, f"got {cdf_mean:.8f}")

    # ------------------------------------------------------------------
    # 7. PDF ≈ finite-difference of CDF
    # ------------------------------------------------------------------
    section("7. PDF–CDF finite-difference consistency")

    cdf_plus = calculate_cdf(X + EPSILON, P)
    cdf_minus = calculate_cdf(X - EPSILON, P)
    fd_pdf = (cdf_plus - cdf_minus) / (2.0 * EPSILON)

    print(f"  PDF from integration : {pdf_val:.8f}")
    print(f"  PDF from finite diff : {fd_pdf:.8f}")

    rel_err = abs(pdf_val - fd_pdf) / max(abs(fd_pdf), 1e-12)
    check(
        "PDF matches finite-difference CDF to 0.1 %",
        rel_err < 1e-3,
        f"rel_err={rel_err:.2e}",
    )

    # ------------------------------------------------------------------
    # 8. Halley solver round-trip  F(Q(u)) ≈ u
    # ------------------------------------------------------------------
    section("8. Halley solver round-trip")

    for u in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        x_star = run_halley_solver(u, P)
        cdf_check = calculate_cdf(x_star, P)
        err = abs(cdf_check - u)
        print(f"  u={u:.2f}  x*={x_star:.6f}  F(x*)={cdf_check:.8f}  err={err:.2e}")
        check(f"Halley: |F(Q({u})) - {u}| < 1e-6", err < 1e-6, f"err={err:.2e}")

    # ------------------------------------------------------------------
    # 9. Brent solver round-trip  (reference cross-check)
    # ------------------------------------------------------------------
    section("9. Brent solver round-trip")

    for u in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        x_star = run_brent_solver(u, P)
        cdf_check = calculate_cdf(x_star, P)
        err = abs(cdf_check - u)
        print(f"  u={u:.2f}  x*={x_star:.6f}  F(x*)={cdf_check:.8f}  err={err:.2e}")
        check(f"Brent:  |F(Q({u})) - {u}| < 1e-6", err < 1e-6, f"err={err:.2e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("Summary")
    n_fail = len(check._failures)
    if n_fail == 0:
        print("  All checks passed.")
    else:
        print(f"  {n_fail} check(s) FAILED:")
        for label in check._failures:
            print(f"    - {label}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_solvers()

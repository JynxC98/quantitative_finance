"""
test_solvers.py
---------------
Test suite for Gil-Pelaez CDF inversion and quantile solvers.

Runs as a plain Python script (no pytest required), but is also
fully compatible with pytest when available.

Test groups
-----------
1. Integrand spot checks       — sign/finiteness/self-consistency of
                                  cdf/pdf/d_pdf integrands
2. CDF structural properties   — monotonicity, boundary behaviour, [0,1] range
3. PDF vs finite difference    — f(x) ~ [F(x+eps) - F(x-eps)] / 2eps
4. Bisection solver round-trips — F(Q(p)) = p for p in (0,1), cross-checked
                                  against scipy.optimize.brentq
5. Halley solver                — agrees with bisection and with brentq
6. invert_variance_cdf dispatch — routes tail probabilities to bisection
7. calculate_integral interface — returns float, consistent with calculate_cdf
8. Edge cases                   — invalid inputs raise cleanly

Reference values are NOT hard-coded. Independent ground truth is computed
at runtime using scipy.integrate.quad (adaptive quadrature, for CDF/PDF)
and scipy.optimize.brentq (for quantiles), so these tests stay valid even
if internal parameters like damp, TAIL_EPS, or node counts change.

Author: Harsh Parikh
"""

import sys
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from bk_helpers.char_func import HestonParams
from bk_helpers.solvers import (
    cdf_integrand,
    pdf_integrand,
    d_pdf_integrand,
    calculate_cdf,
    calculate_pdf,
    calculate_integral,
    run_bisection_solver,
    run_halley_solver,
    invert_variance_cdf,
    TAIL_EPS,
)

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_results = {"passed": 0, "failed": 0}


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  [PASS] {label}")
        _results["passed"] += 1
    else:
        print(f"  [FAIL] {label}  {detail}")
        _results["failed"] += 1
    return condition


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

# Standard Heston parameters satisfying the Feller condition:
#   2*kappa*theta = 0.16 > sigma^2 = 0.09
P = HestonParams(
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    v_t=0.04,
    v_u=0.04,
    dt=1.0,
)

MEAN_VAR = 0.5 * (P.v_t + P.v_u) * P.dt  # trapezoidal approximation


def reference_cdf(x: float, p: HestonParams, upper: float = 2000.0) -> float:
    """
    Independent ground-truth CDF via adaptive quadrature (scipy.quad)
    over the same cdf_integrand your solver uses, rather than the
    composite Gauss-Legendre scheme in calculate_integral. This cross-
    checks the *quadrature method*, not just re-runs the same code path.
    """
    val, _ = quad(lambda u: cdf_integrand(x, u, p), 0.0, upper, limit=200)
    return 0.5 + val


def reference_pdf(x: float, p: HestonParams, upper: float = 2000.0) -> float:
    """Independent ground-truth PDF via adaptive quadrature."""
    val, _ = quad(lambda u: pdf_integrand(x, u, p), 0.0, upper, limit=200)
    return val


def reference_quantile(var: float, p: HestonParams) -> float:
    """
    Independent root-finder (scipy.optimize.brentq) for cross-checking
    run_bisection_solver / run_halley_solver, bracketing around the
    same CIR-mean heuristic used internally.
    """
    kappa, theta, dt, v0 = p.kappa, p.theta, p.dt, p.v_t
    if abs(kappa) < 1e-10:
        mean = v0 * dt
    else:
        mean = theta * dt + (v0 - theta) * (1.0 - np.exp(-kappa * dt)) / kappa

    lo, hi = max(mean * 1e-3, 1e-9), mean * 10.0
    f_lo = calculate_cdf(lo, p) - var
    f_hi = calculate_cdf(hi, p) - var
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 50:
        lo /= 2.0
        hi *= 2.0
        f_lo = calculate_cdf(lo, p) - var
        f_hi = calculate_cdf(hi, p) - var
        expand += 1

    return brentq(lambda x: calculate_cdf(x, p) - var, lo, hi, xtol=1e-10, maxiter=200)


# ---------------------------------------------------------------------------
# 1. Integrand spot checks
# ---------------------------------------------------------------------------


def test_integrands():
    section("1. Integrand spot checks")

    cdf_val = cdf_integrand(MEAN_VAR, 1.0, P)
    dpdf_val = d_pdf_integrand(MEAN_VAR, 1.0, P)

    # Identity that holds for ALL parameters at u=1, by construction:
    #   cdf_integrand = -Im[...] / (u*pi),  d_pdf_integrand = u*Im[...]/pi
    # so at u=1 they are exact negatives of each other -- no magic number.
    check(
        "d_pdf_integrand(mean,1) == -cdf_integrand(mean,1)  [at u=1]",
        abs(dpdf_val + cdf_val) < 1e-12,
        f"d_pdf={dpdf_val:.6e}, -cdf={-cdf_val:.6e}",
    )

    # Analytic singularity limits (derived directly from the formulas,
    # not fitted to a particular parameter set)
    x_test = 0.005
    sing_cdf = cdf_integrand(x_test, 1e-10, P)
    check(
        "cdf_integrand singularity u->0 returns x/pi",
        abs(sing_cdf - x_test / np.pi) < 1e-10,
        f"got {sing_cdf:.8f}, expected {x_test/np.pi:.8f}",
    )

    sing_pdf = pdf_integrand(x_test, 1e-10, P)
    check(
        "pdf_integrand singularity u->0 returns 1/pi",
        abs(sing_pdf - 1.0 / np.pi) < 1e-10,
        f"got {sing_pdf:.8f}",
    )

    sing_dpdf = d_pdf_integrand(x_test, 1e-10, P)
    check(
        "d_pdf_integrand singularity u->0 returns 0",
        abs(sing_dpdf) < 1e-10,
        f"got {sing_dpdf:.8e}",
    )

    u_vals = [0.0, 1e-9, 0.1, 1.0, 10.0, 100.0, 1000.0]
    x_test = 0.003
    all_finite = all(
        np.isfinite(cdf_integrand(x_test, u, P))
        and np.isfinite(pdf_integrand(x_test, u, P))
        and np.isfinite(d_pdf_integrand(x_test, u, P))
        for u in u_vals
    )
    check("All integrands finite for u in [0, 1000]", all_finite)


# ---------------------------------------------------------------------------
# 2. CDF structural properties
# ---------------------------------------------------------------------------


def test_cdf():
    section("2. CDF structural properties")

    cdf_low = calculate_cdf(1e-4, P)
    check("F(near-zero) is small", cdf_low < 0.05, f"got {cdf_low:.6f}")

    x_vals = np.linspace(1e-4, 0.02, 12)
    cdfs = [calculate_cdf(x, P) for x in x_vals]
    non_mono = [
        (x_vals[i], cdfs[i], cdfs[i + 1])
        for i in range(len(cdfs) - 1)
        if cdfs[i] >= cdfs[i + 1]
    ]
    check(
        "F strictly monotone increasing", len(non_mono) == 0, f"violated at: {non_mono}"
    )

    out_of_range = [
        (x, c) for x, c in zip(x_vals, cdfs) if not (-1e-4 <= c <= 1 + 1e-4)
    ]
    check(
        "F(x) in [0,1] across support",
        len(out_of_range) == 0,
        f"violations: {out_of_range}",
    )

    # Cross-check against an independent adaptive-quadrature implementation
    # of the same Gil-Pelaez formula, instead of a hard-coded expected value.
    for x in [0.002, 0.0033, 0.005, 0.008]:
        got = calculate_cdf(x, P)
        ref = reference_cdf(x, P)
        check(
            f"F({x}) matches independent quad reference",
            abs(got - ref) < 1e-4,
            f"got {got:.6f}, reference {ref:.6f}",
        )


# ---------------------------------------------------------------------------
# 3. PDF vs finite difference
# ---------------------------------------------------------------------------


def test_pdf():
    section("3. PDF vs finite difference  [f(x) ~ dF/dx]")

    epsilon = 1e-5
    for x in [0.002, 0.003, 0.0033, 0.004, 0.005]:
        pdf_val = calculate_pdf(x, P)
        fd = (calculate_cdf(x + epsilon, P) - calculate_cdf(x - epsilon, P)) / (
            2 * epsilon
        )
        rel_err = abs(pdf_val - fd) / max(abs(fd), 1e-10)
        check(
            f"PDF({x:.4f}) matches finite difference to <1%",
            rel_err < 0.01,
            f"PDF={pdf_val:.4f}, FD={fd:.4f}, rel_err={rel_err:.2e}",
        )

    # Independent cross-check against adaptive quadrature
    for x in [0.002, 0.0033, 0.006]:
        got = calculate_pdf(x, P)
        ref = reference_pdf(x, P)
        check(
            f"PDF({x}) matches independent quad reference",
            abs(got - ref) < 1e-4,
            f"got {got:.6f}, reference {ref:.6f}",
        )

    x_grid = np.linspace(0.001, 0.02, 20)
    pdf_vals = [(x, calculate_pdf(x, P)) for x in x_grid]
    neg = [(x, v) for x, v in pdf_vals if v < -1e-3]
    check("PDF(x) >= 0 across support", len(neg) == 0, f"negative at: {neg}")


# ---------------------------------------------------------------------------
# 4. Bisection solver
# ---------------------------------------------------------------------------


def test_bisection():
    section("4. Bisection solver:  F(Q(p)) = p")

    probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    quantiles = []
    for var in probs:
        x_star = run_bisection_solver(var, P)
        err = abs(calculate_cdf(x_star, P) - var)
        check(f"Bisection round-trip p={var}", err < 1e-5, f"err={err:.2e}")
        quantiles.append(x_star)

    check(
        "Bisection quantiles strictly monotone",
        all(quantiles[i] < quantiles[i + 1] for i in range(len(quantiles) - 1)),
    )
    check("All bisection quantiles > 0", all(q > 0 for q in quantiles))

    # Cross-check against an independent root-finder (brentq), rather
    # than fixed reference numbers.
    for var in probs:
        got = run_bisection_solver(var, P)
        ref = reference_quantile(var, P)
        check(
            f"Bisection Q({var}) matches brentq reference",
            abs(got - ref) < 1e-5,
            f"got {got:.8f}, reference {ref:.8f}",
        )

    for bad in [0.0, 1.0, -0.1, 1.1]:
        try:
            run_bisection_solver(bad, P)
            check(
                f"Bisection raises ValueError for var={bad}",
                False,
                "no exception raised",
            )
        except ValueError:
            check(f"Bisection raises ValueError for var={bad}", True)
        except Exception as e:
            check(
                f"Bisection raises ValueError for var={bad}",
                False,
                f"got {type(e).__name__}",
            )


# ---------------------------------------------------------------------------
# 5. Halley solver
# ---------------------------------------------------------------------------


def test_halley():
    section("5. Halley solver:  agrees with bisection and brentq")

    for var in [0.10, 0.25, 0.50, 0.75, 0.90]:
        x_halley = run_halley_solver(var, P)
        x_bisect = run_bisection_solver(var, P)
        diff = abs(x_halley - x_bisect)
        check(f"Halley ~ bisection at p={var}", diff < 1e-6, f"diff={diff:.2e}")

        x_ref = reference_quantile(var, P)
        diff_ref = abs(x_halley - x_ref)
        check(
            f"Halley ~ brentq reference at p={var}",
            diff_ref < 1e-5,
            f"diff={diff_ref:.2e}",
        )

    for var in [0.10, 0.50, 0.90]:
        x_star = run_halley_solver(var, P)
        err = abs(calculate_cdf(x_star, P) - var)
        check(f"Halley round-trip p={var}", err < 1e-5, f"err={err:.2e}")

    for bad in [0.0, 1.0]:
        try:
            run_halley_solver(bad, P)
            check(
                f"Halley raises ValueError for var={bad}", False, "no exception raised"
            )
        except ValueError:
            check(f"Halley raises ValueError for var={bad}", True)
        except Exception as e:
            check(
                f"Halley raises ValueError for var={bad}",
                False,
                f"got {type(e).__name__}",
            )


# ---------------------------------------------------------------------------
# 6. invert_variance_cdf dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher():
    section("6. invert_variance_cdf: tail vs. mid-range dispatch")

    # Mid-range probability should agree with Halley
    mid_var = 0.5
    dispatched = invert_variance_cdf(mid_var, P)
    halley_direct = run_halley_solver(mid_var, P)
    check(
        "Mid-range dispatch matches Halley",
        abs(dispatched - halley_direct) < 1e-9,
        f"dispatched={dispatched:.8f}, halley={halley_direct:.8f}",
    )

    # Tail probability should agree with bisection, and still round-trip
    tail_var = TAIL_EPS / 2.0
    dispatched_tail = invert_variance_cdf(tail_var, P)
    bisection_direct = run_bisection_solver(tail_var, P)
    check(
        "Tail dispatch matches bisection",
        abs(dispatched_tail - bisection_direct) < 1e-9,
        f"dispatched={dispatched_tail:.8f}, bisection={bisection_direct:.8f}",
    )
    err = abs(calculate_cdf(dispatched_tail, P) - tail_var)
    check(f"Tail dispatch round-trip p={tail_var}", err < 1e-4, f"err={err:.2e}")

    for bad in [0.0, 1.0, -0.1, 1.1]:
        try:
            invert_variance_cdf(bad, P)
            check(
                f"Dispatcher raises ValueError for var={bad}",
                False,
                "no exception raised",
            )
        except ValueError:
            check(f"Dispatcher raises ValueError for var={bad}", True)
        except Exception as e:
            check(
                f"Dispatcher raises ValueError for var={bad}",
                False,
                f"got {type(e).__name__}",
            )


# ---------------------------------------------------------------------------
# 7. calculate_integral interface
# ---------------------------------------------------------------------------


def test_calculate_integral():
    section("7. calculate_integral interface")

    result = calculate_integral(pdf_integrand, MEAN_VAR, P)
    check(
        "calculate_integral returns float",
        isinstance(result, float),
        f"got {type(result)}",
    )
    check("PDF integral > 0 at mean", result > 0, f"got {result}")

    integral = calculate_integral(cdf_integrand, MEAN_VAR, P)
    direct = calculate_cdf(MEAN_VAR, P)
    check(
        "calculate_cdf == 0.5 + calculate_integral(cdf_integrand)",
        abs((0.5 + integral) - direct) < 1e-10,
        f"0.5+integral={0.5+integral:.10f}, direct={direct:.10f}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"\nHeston params: kappa={P.kappa}, theta={P.theta}, sigma={P.sigma}, "
        f"V_t={P.v_t}, V_u={P.v_u}, dt={P.dt:.4f}"
    )
    print(
        f"Feller condition: 2*kappa*theta={2*P.kappa*P.theta:.3f} > sigma^2={P.sigma**2:.3f}  "
        f"{'OK' if 2*P.kappa*P.theta > P.sigma**2 else 'VIOLATED'}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_integrands()
        test_cdf()
        test_pdf()
        test_bisection()
        test_halley()
        test_dispatcher()
        test_calculate_integral()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'='*55}")
    print(f"  {_results['passed']}/{total} passed   |   {_results['failed']} failed")
    print(f"{'='*55}\n")
    sys.exit(0 if _results["failed"] == 0 else 1)

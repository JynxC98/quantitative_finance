/**
 * @file fourier_implementation.hpp
 * @brief FFT-based construction of the integrated-variance CDF tables used
 * by the Broadie-Kaya exact simulation scheme.
 *
 * @details
 * cdf_table.hpp::buildCDFTable computes each table entry via an independent
 * adaptive Gauss-Legendre quadrature over the Gil-Pelaez frequency integral
 * (solvers.hpp): every single (v_t, x) pair re-derives the truncation
 * frequency (findCriticalfreq) and re-walks the whole panel structure from
 * scratch. That's ~12ms per point -- for a 20-node v_t grid at 50
 * points/table that's already ~12 seconds, and it scales linearly with
 * both knobs. That cost is why the v_t grid stayed coarse (n_v=20): a
 * convexity analysis of E[int_var | v_t] showed the resulting linear
 * quantile-interpolation between grid nodes is a chord across a genuinely
 * convex curve, systematically overshooting the true conditional mean --
 * a real bias that shrinks with the square of the bracket width, so it
 * needed a much finer v_t grid to run down, which the quadrature approach
 * couldn't afford.
 *
 * This file replaces the per-point evaluation with one FFT per v_t node:
 * computeCDFGrid() (fourier_transform.cpp) evaluates the characteristic
 * function once per frequency and inverts the WHOLE CDF curve in a single
 * O(N log N) transform, instead of re-deriving it independently for every
 * x. Measured on the paper's benchmark parameters: a 65536-point CDF grid
 * computes in ~0.06s -- roughly 14,000x more CDF points per second than
 * the quadrature path, which is what makes a much denser v_t grid (and
 * much denser per-table x-grid) affordable.
 *
 * ── The Nyquist subtlety ─────────────────────────────────────────────────
 * A naive port -- reusing critical_freq, the frequency at which
 * calculateCDF's tolerance check stops the quadrature -- undershoots
 * badly. With an FFT of size N and frequency step du:
 *
 *     dx    = 2*pi / (N * du)     (x-grid spacing, i.e. resolution)
 *     x_max = 2*pi / du           (x-grid range covered)
 *
 * Notice x_max depends only on du, and dx depends on the *total* frequency
 * range N*du -- neither is about whether |phi(u)| has decayed to
 * negligible size. The integrated-variance CDF rises from 0 to ~0.95
 * within a very narrow x-window (order 0.03 for the paper's benchmark
 * parameters), and resolving a feature that steep needs broadband
 * frequency content (the Fourier uncertainty principle) far beyond the
 * point where |phi(u)| stops mattering to a tolerance-based quadrature
 * cutoff: critical_freq ~1900 there, versus the ~12500 actually needed to
 * hit dx ~0.0002. Reusing critical_freq as the FFT's frequency range gives
 * plausible-looking but measurably wrong CDF values near the steep region
 * (confirmed empirically: up to ~7% absolute error near the rise at
 * critical_freq-based tuning, versus <0.3% -- shrinking further with N --
 * once tuned for x-resolution instead of CF decay).
 *
 * ── du has its own, separate accuracy floor ─────────────────────────────
 * Satisfying x_max and dx alone is not sufficient. computeCDFGrid's DFT
 * approximates the Gil-Pelaez integral with a plain Riemann sum in u (no
 * adaptive per-panel refinement the way calculateIntegral's Gauss-Legendre
 * panels have), and it is implicitly periodic in x with period x_max. A du
 * that is too coarse leaves this Riemann sum under-resolved right where
 * the CDF is steepest -- near x=0, which sits at the seam of that
 * periodicity -- producing values that are wrong (too high) rather than
 * ~0. Because that corrupted region is at the *start* of the table, the
 * monotonicity fix buildCDFTableFFT applies (a running max, mirroring
 * buildCDFTable's own fix for the quadrature path's oscillatory noise)
 * then locks the wrong high value in as a floor and holds it flat until
 * the true curve grows past it -- silently flattening exactly the region
 * holding most of the probability mass. This is NOT fixed by increasing
 * N*du (more x-coverage) or by increasing N alone at fixed N*du (finer
 * dx): it is governed by du by itself. Empirically (paper benchmark
 * parameters, N*du held at ~12566): du=0.77 leaves cdf(0+) ~0.002, du=0.19
 * leaves ~0.0005, du=0.048 leaves ~0.0001 -- roughly linear in du. A du
 * that instead grows to ~4.6 (as a naive x_max_needed-only derivation of
 * du produces once x_max_needed is small, since du = 2*pi/x_max_needed)
 * leaves the CDF pinned near ~0.01 through several hundredths of the
 * x-range, corrupting far more of the table than the modest-looking du
 * value would suggest -- confirmed to break the full pipeline (put price
 * off by ~70%) before this was found.
 *
 * chooseFFTGridParams() therefore enforces max_du as an independent,
 * empirically-grounded ceiling on du, alongside (not in place of) the
 * x_max/dx coverage requirement -- decoupled from critical_freq entirely.
 *
 * @author Harsh Parikh
 */

#if !defined(FOURIER_IMPLEMENTATION_HPP)
#define FOURIER_IMPLEMENTATION_HPP

#include <vector>

#include "heston_params.hpp"
#include "cdf_table.hpp"

/**
 * @brief FFT frequency-grid parameters for computeCDFGrid().
 *
 * @param N   FFT size. Always a power of 2 (required by the radix-2
 *            recursive FFT in fourier_transform.cpp).
 * @param du  Frequency-domain step (spacing between successive u samples
 *            of the characteristic function).
 */
struct FFTGridParams
{
    int N;
    double du;
};

/**
 * @brief Chooses (N, du) so the resulting x-grid spacing is at most
 * target_dx, the x-grid range covers at least x_max_needed, AND du itself
 * is at most max_du (the near-x=0 accuracy floor -- see the file-level doc
 * above for why this is a separate requirement from the other two).
 *
 * du is first set to the smaller of what x_max_needed alone would require
 * (2*pi/x_max_needed) and max_du -- in practice max_du is the binding
 * constraint whenever x_max_needed is modest, which pushes x_max well
 * past what's strictly needed, but that is harmless (just some unused
 * table range beyond where the CDF has already saturated to 1). N is then
 * chosen (and rounded up to the next power of 2) to make
 * dx = 2*pi/(N*du) at least as fine as target_dx; rounding N up only
 * makes dx finer than requested and does not change x_max, since
 * x_max = 2*pi/du depends on du alone.
 *
 * @param x_max_needed  Minimum x-range the grid must cover (e.g. a safety
 *                       margin over the CDF table's upper bound u_eps).
 * @param target_dx     Desired (maximum) x-grid spacing.
 * @param max_du        Ceiling on the frequency step, independent of the
 *                       x_max_needed/target_dx requirements -- see the
 *                       file-level doc for the empirical grounding.
 * @return               FFTGridParams with N rounded up to a power of 2.
 */
FFTGridParams chooseFFTGridParams(double x_max_needed, double target_dx, double max_du);

/**
 * @brief Builds a single CDFTable for the given (v_u, v_t) via FFT-based
 * Gil-Pelaez inversion (computeCDFGrid), in place of the direct per-point
 * adaptive-quadrature approach in cdf_table.hpp::buildCDFTable.
 *
 * Drop-in replacement for buildCDFTable: same CDFTable output type and the
 * same downstream usage (sampleFromTable), just ~10,000x cheaper to build,
 * which is what lets callers afford far denser tables and v_t grids.
 *
 * @param p      Heston parameters; must have v_u, v_t, dt, sigma set.
 * @param v_min  Lower bound used for the table's first x-grid point
 *               (matches buildCDFTable's convention).
 * @param N      FFT size (must be a power of 2 -- see FFTGridParams).
 * @param du     Frequency-domain step for computeCDFGrid.
 * @return       CDFTable with an N-point x_grid/cdf_vals pair, monotonic
 *               and clamped to [0, 1] (the same fix buildCDFTable applies,
 *               since the truncated, damped Gil-Pelaez integral still
 *               leaves small oscillatory noise even after FFT inversion).
 */
CDFTable buildCDFTableFFT(const HestonParams &p, double v_min, int N, double du);

/**
 * @brief Builds the full v_t-indexed grid of CDFTables via FFT, for every
 * node in v_t_nodes, sharing one set of FFT parameters chosen from the
 * grid's worst case.
 *
 * All nodes share (N, du) rather than each choosing its own: the x-window
 * width (std1 in buildCDFTable's mu1/std1 derivation) depends only on
 * v_u, sigma and dt -- all fixed across the v_t grid -- so only the
 * *location* of the window (mu1, which grows with v_t) varies node to
 * node. Sizing the FFT's x-coverage off the largest v_t node in the grid
 * is therefore enough to cover every node with margin, and keeps every
 * table's grid spacing/layout uniform.
 *
 * @param p          Heston parameters; p.v_u must be set (p.v_t is
 *                    overwritten per node; the input value is ignored).
 * @param v_min       Lower bound passed through to buildCDFTableFFT.
 * @param v_t_nodes   v_t grid nodes to build a table for, in any order.
 * @return            CDFTables in the same order as v_t_nodes.
 */
std::vector<CDFTable> buildCDFTableGridFFT(const HestonParams &p, double v_min,
                                            const std::vector<double> &v_t_nodes);

#endif

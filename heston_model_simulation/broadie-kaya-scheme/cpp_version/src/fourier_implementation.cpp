/**
 * @brief FFT-based construction of the integrated-variance CDF tables.
 * See helpers/fourier_implementation.hpp for the full rationale.
 *
 * @author Harsh Parikh
 */

#include <cmath>
#include <algorithm>

#include "../helpers/fourier_implementation.hpp"
#include "../helpers/fourier_transform.hpp"

FFTGridParams chooseFFTGridParams(double x_max_needed, double target_dx, double max_du)
{
    // x_max = 2*pi/du depends on du alone; pick the smaller of what
    // x_max_needed alone would require and the max_du accuracy ceiling
    // (see file doc -- max_du is the binding constraint whenever
    // x_max_needed is modest, which is the normal case here).
    double du = std::min(2.0 * M_PI / x_max_needed, max_du);

    // dx = 2*pi/(N*du) -- solve for the total frequency range N*du needed
    // to hit target_dx, then divide by du to get N.
    double u_max = 2.0 * M_PI / target_dx;
    int n_needed = static_cast<int>(std::ceil(u_max / du));

    int N = 1;
    while (N < n_needed)
        N <<= 1;

    return {N, du};
}

CDFTable buildCDFTableFFT(const HestonParams &p, double v_min, int N, double du)
{
    CDFGrid grid = computeCDFGrid(p, N, du);

    CDFTable table;
    table.v_u = p.v_u;
    table.v_t = p.v_t;
    table.x_grid.resize(N);
    table.cdf_vals.resize(N);

    for (int i = 0; i < N; ++i)
    {
        table.x_grid[i] = std::max(v_min, i * grid.dx);
        table.cdf_vals[i] = grid.cdf[i];
    }

    // Same fix as buildCDFTable: the truncated, damped Gil-Pelaez integral
    // leaves small oscillatory noise even after FFT inversion, which is
    // not guaranteed monotonic. sampleFromTable's binary search assumes a
    // sorted cdf_vals, so enforce a running maximum before this table is
    // used for quantile lookup.
    for (int i = 1; i < N; ++i)
        table.cdf_vals[i] = std::max(table.cdf_vals[i], table.cdf_vals[i - 1]);
    for (int i = 0; i < N; ++i)
        table.cdf_vals[i] = std::min(1.0, std::max(0.0, table.cdf_vals[i]));

    return table;
}

std::vector<CDFTable> buildCDFTableGridFFT(const HestonParams &p, double v_min,
                                            const std::vector<double> &v_t_nodes)
{
    // Worst-case u_eps across the grid: mu1 grows with v_t, std1 does not
    // (it depends only on v_u, sigma, dt -- all fixed across the grid), so
    // the largest v_t node sets how far out in x the FFT needs to reach.
    double v_t_max_node = *std::max_element(v_t_nodes.begin(), v_t_nodes.end());
    double mu1 = 0.5 * (p.v_u + v_t_max_node) * p.dt;
    double var = p.sigma * p.sigma * p.v_u * p.dt * p.dt / 2.0;
    double std1 = std::sqrt(std::max(var, 0.0));
    double u_eps_max = mu1 + 6.0 * std1;

    // 3x margin over the worst-case tail bound, a fine target x-spacing,
    // and max_du=0.2 -- empirically, du=0.19 left the near-x=0 floor at
    // ~0.0005 (see file doc), small enough not to visibly distort the
    // sampled distribution while keeping N (and table-build time) modest.
    FFTGridParams fft = chooseFFTGridParams(3.0 * u_eps_max, 2e-4, 0.2);

    std::vector<CDFTable> tables(v_t_nodes.size());
    HestonParams temp = p;
    for (size_t j = 0; j < v_t_nodes.size(); ++j)
    {
        temp.v_t = v_t_nodes[j];
        tables[j] = buildCDFTableFFT(temp, v_min, fft.N, fft.du);
    }
    return tables;
}

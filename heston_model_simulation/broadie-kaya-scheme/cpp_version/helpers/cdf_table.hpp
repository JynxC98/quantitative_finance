/**
 * @brief This header file stores the implementation of the caching of
 * char function values across different levels of variances.
 */

#ifndef CDF_TABLE_HPP
#define CDF_TABLE_HPP

#include <vector>
#include <filesystem>
#include <string>
#include <fstream>
#include "heston_params.hpp"
#include "solvers.hpp"
#include "helpers.hpp"

/**
 * @brief Precomputed CDF lookup table for a fixed (v_u, v_t) pair.
 *
 * Used to replace per-sample Halley's method quantile solves at runtime
 * via linear interpolation over a prebuilt x -> CDF(x) mapping.
 */
struct CDFTable
{
    std::vector<double> x_grid;   ///< Uniformly spaced variance values.
    std::vector<double> cdf_vals; ///< CDF evaluated at each x_grid point.
    double v_u;                   ///< Variance at the start of the timestep.
    double v_t;                   ///< Variance at the end of the timestep.
};

/**
 * @brief Builds a CDFTable for the given Heston parameters.
 *
 * The upper bound u_eps is set to mu + 10*sigma of a Gaussian approximation
 * to the integrated variance distribution, ensuring the table covers
 * virtually all of the probability mass.
 *
 * @param p        Heston model parameters; must have v_u, v_t, dt, sigma set.
 * @param n_points Number of grid points (default: 100).
 * @return         Populated CDFTable over [1e-10, u_eps].
 */
inline CDFTable buildCDFTable(const HestonParams &p, int n_points = 100)
{
    double mu1 = 0.5 * (p.v_u + p.v_t) * p.dt;
    double var = p.sigma * p.sigma * p.v_u * p.dt * p.dt / 2.0;
    double std1 = std::sqrt(std::max(var, 0.0));
    double u_eps = mu1 + 10.0 * std1;
    u_eps = std::max(u_eps, 1e-6);

    CDFTable table;
    table.v_u = p.v_u;
    table.v_t = p.v_t;
    table.x_grid = getLinspace(1e-10, u_eps, n_points);
    table.cdf_vals.resize(n_points);

    for (int i = 0; i < n_points; ++i)
        table.cdf_vals[i] = calculateCDF(table.x_grid[i], p);

    return table;
}

/**
 * @brief Inverts the CDF table at quantile U via linear interpolation.
 *
 * @param U     Uniform quantile in [0, 1].
 * @param table Precomputed CDFTable to invert.
 * @return      Variance sample corresponding to quantile U.
 * @note        Extrapolation behaviour for U outside [0, 1] is inherited
 *              from linear_interpolate.
 */
inline double sampleFromTable(double U, const CDFTable &table)
{
    return linear_interpolate(U, table.cdf_vals, table.x_grid);
}
/**
 * @brief Serializes a 2D grid of CDFTable objects to a binary file.
 *
 * Layout: [n_v | n_points | v_u | v_t | x_grid... | cdf_vals...] per cell,
 * written in row-major order.
 *
 * @param tables    2D grid of CDFTable objects, assumed square (n_v x n_v).
 * @param path      Output file path.
 *
 * @warning The cache is only valid for the exact (p.theta, n_v, n_points)
 *          combination used during construction. No header validation is
 *          performed on load — mismatched parameters will silently produce
 *          incorrect results. Delete the cache file whenever any of these
 *          change.
 */
inline void saveCDFTableGrid(const std::vector<std::vector<CDFTable>> &tables,
                             const std::string &path)
{
    std::ofstream f(path, std::ios::binary);
    size_t n_v = tables.size();
    f.write(reinterpret_cast<const char *>(&n_v), sizeof(n_v));

    for (const auto &row : tables)
        for (const auto &table : row)
        {
            size_t n = table.x_grid.size();
            f.write(reinterpret_cast<const char *>(&n), sizeof(n));
            f.write(reinterpret_cast<const char *>(&table.v_u), sizeof(double));
            f.write(reinterpret_cast<const char *>(&table.v_t), sizeof(double));
            f.write(reinterpret_cast<const char *>(table.x_grid.data()), n * sizeof(double));
            f.write(reinterpret_cast<const char *>(table.cdf_vals.data()), n * sizeof(double));
        }
}

/**
 * @brief Deserializes a 2D grid of CDFTable objects from a binary file
 *        previously written by saveCDFTableGrid().
 *
 * Resizes @p tables to match the stored grid dimensions before populating.
 *
 * @param tables    Output grid, resized and populated in place.
 * @param path      Input file path.
 *
 * @warning No version or parameter validation is performed. Loading a cache
 *          built under different (p.theta, n_v, n_points) parameters will
 *          silently corrupt the simulation. The caller is responsible for
 *          ensuring cache validity.
 */
inline void loadCDFTableGrid(std::vector<std::vector<CDFTable>> &tables,
                             const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    size_t n_v;
    f.read(reinterpret_cast<char *>(&n_v), sizeof(n_v));
    tables.assign(n_v, std::vector<CDFTable>(n_v));

    for (auto &row : tables)
        for (auto &table : row)
        {
            size_t n;
            f.read(reinterpret_cast<char *>(&n), sizeof(n));
            f.read(reinterpret_cast<char *>(&table.v_u), sizeof(double));
            f.read(reinterpret_cast<char *>(&table.v_t), sizeof(double));
            table.x_grid.resize(n);
            table.cdf_vals.resize(n);
            f.read(reinterpret_cast<char *>(table.x_grid.data()), n * sizeof(double));
            f.read(reinterpret_cast<char *>(table.cdf_vals.data()), n * sizeof(double));
        }
}
#endif
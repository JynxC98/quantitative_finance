/**
 * @brief Thiis header file stores the structure of the Heson Model's
 * input parameters.
 */

#pragma once

/**
 * @brief This struct format acts as a container for Heston parameters
 *
 * @param kappa: The mean reversion rate
 * @param theta: The long run average
 * @param sigma: Volatility of variance
 * @param v_u: The variance value at timestep u
 * @param v_t: The variance value at timestep t (t > u)
 * @param dt: The value of timestep, t-u
 * @param v0: The initial value of variance
 * @param rho: The correlation between the asset price and the variance process.
 */
struct HestonParams
{
    double kappa;
    double theta;
    double sigma;
    double v_u;
    double v_t;
    double dt;
    double v0;
    double rho;
};
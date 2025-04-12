/**
 * @file heston_analytical_solution.cpp
 * @brief Analytical pricing of Geometric Asian Options under Heston's Stochastic Volatility Model.
 *
 * This module implements the closed-form pricing approach for geometric Asian options
 * under the Heston stochastic volatility framework, as proposed in the paper:
 *
 * Kim, B., & Wee, I. S. (2011). Pricing of geometric Asian options under Heston's
 * stochastic volatility model. *Quantitative Finance*, 11(12), 1795–1811.
 * https://doi.org/10.1080/14697688.2011.596844
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "helper_functions.h"

using namespace std;

/**
 * @brief Calculate the price of a geometric Asian call option under Heston's stochastic volatility model.
 *
 * This class implements the semi-analytical pricing of a geometric Asian call option
 * based on the Heston stochastic volatility framework.
 *
 * @param S0 Initial stock price.
 * @param v0 Initial variance (volatility squared).
 * @param theta Long-term mean of variance.
 * @param sigma Volatility of variance (vol of vol).
 * @param kappa Mean reversion rate of variance.
 * @param rho Correlation between the asset price and variance.
 * @param r Risk-free interest rate.
 * @param n Number of terms in the series expansion (for approximation).
 * @param T Time to maturity.
 * @param K Strike price.
 * @param int_upper_limit Optional upper limit for integration (default is infinity).
 *
 * @return The price of the geometric Asian call option.
 */
class HestonPricer
{
private:
    double S0;
    double v0;
    double theta;
    double sigma;
    double kappa;
    double rho;
    double r;
    double n;
    double T;
    double K;
    int upper_limit;

public:
    HestonPricer(double S0, double v0, double theta, double sigma, double kappa,
                 double rho, double r, double n, double T,
                 double K, int upper_limit)
    {
        this->S0 = S0;
        this->v0 = v0;
        this->theta = theta;
        this->sigma = sigma;
        this->kappa = kappa;
        this->rho = rho;
        this->r = r;
        this->n = n;
        this->T = T;
        this->K = K;
        this->upper_limit = upper_limit;
    }

    /**
     * @brief Calculate the characteristic function `psi` as defined in the paper
     *
     * @param s: Complex argument for the characteristic function
     * @param w: Second argument for the characteristic function
     *
     * @return The value of the characteristic function `psi`
     */
    complex<double> calculatePSI(complex<double> s, double w);

    /**
     * @brief Calculate the integrand component of the geometric Asian option price.
     *
     * @return Value of the integrand
     */
    double calculateIntegrand();

    /**
     * @brief Calculate the integral compoment of the geometric Asian option price.
     *
     * @return Value of the integral
     */
    double calculateIntegral();

    /**
     * @brief Calculat the price of a geometric Asian call option under Heston's
     * stochastic vol model.
     *
     * @return Price of the geometric Asian call option
     */
};

complex<double> HestonPricer::calculatePSI(complex<double> s, double w)
{
    // The `a_i` terms are the components of the main formula
    double a1 = 2 * v0 / (pow(sigma, 2));
    double a2 = 2 * kappa * theta / (pow(sigma, 2));
    double a3 = log(S0) + ((r * sigma - kappa * theta * rho) * T) / (2 * sigma) - (rho * v0) / sigma;
    double a4 = log(S0) - (rho * v0 / sigma) + (r - rho * kappa * theta / sigma) * T;
    double a5 = (kappa * v0 + pow(kappa, 2) * theta * T) / (pow(sigma, 2));

    // This matrix stores the values mentioned in the paper.
    vector<complex<double>> h_matrix(n + 3, complex<double>(0.0, 0.0));

    // Populating the values of h_matrix as per the paper
    h_matrix[2] = 1.0;
    h_matrix[3] = T * (kappa - w * rho * sigma) / 2.0;

    // This matrix stores the range of numbers
    auto nmat = getArange(1, n + 1); // vector<int> from 1 to n

    // Calculating the value of `A`
    vector<double> A;
    for (int j = 1; j < n; ++j)
    {
        A.push_back(1.0 / (4.0 * nmat[j] * (nmat[j] - 1)));
    }

    // Calculating the values of B, C, D
    complex<double> B = -pow(s, 2) * pow(sigma, 2) * (1.0 - pow(rho, 2)) * pow(T, 2);

    complex<double> C = T * (s * sigma * T * (sigma - 2.0 * rho * kappa) - 2.0 * s * w * pow(sigma, 2) * T * (1.0 - pow(rho, 2)));

    complex<double> D = T * (pow(kappa, 2) * T - 2.0 * s * rho * sigma - w * (2.0 * rho * kappa - sigma) * sigma * T - pow(w, 2) * (1.0 - pow(rho, 2)) * pow(sigma, 2) * T);

    // Iteratively updating h_matrix as per recurrence
    for (int j = 4; j < n + 3; ++j)
    {
        h_matrix[j] = A[j - 4] * (B * h_matrix[j - 4] + C * h_matrix[j - 3] + D * h_matrix[j - 2]);
    }

    // Calculating H and H_tilde
    complex<double> H(0.0, 0.0);
    for (int j = 2; j < h_matrix.size(); ++j)
    {
        H += h_matrix[j];
    }

    complex<double> H_tilde(0.0, 0.0);
    for (int j = 3; j < h_matrix.size(); ++j)
    {
        H_tilde += (static_cast<double>(j - 2) / T) * h_matrix[j];
    }

    // Returning the final value of the psi function
    return exp(-a1 * (H_tilde / H) - a2 * log(H) + a3 * s + a4 * w + a5);
}

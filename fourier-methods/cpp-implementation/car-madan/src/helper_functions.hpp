/**
 * @file helper_functions.hpp
 * @brief These methods are used to assist the implementations
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>

/**
 * @brief This function creates a linspace between two values
 *
 * @param lower_val: The lower limit
 * @param upper_val: The upper limit
 * @param N: The number of grid points
 */

std::vector<double> getLinSpace(double lower_val, double upper_val, int N)
{
    // Initialising an empty array of gridpoints.
    std::vector<double> grid_points;

    // Base condition (Excluding negative elements for the sake of simplicity)
    if (N <= 1)
    {
        // If N <= 1, just return lower_val
        grid_points.push_back(lower_val);
        return grid_points;
    }

    // Calculating the step size
    double step = (upper_val - lower_val) / (N - 1);

    for (int i = 0; i < N; ++i)
    {
        grid_points.push_back(lower_val + i * step);
    }

    return grid_points;
}

/**
 * @brief This function calculates the next greatest power of 2 using the
 * bitwise left shift operator.
 */
int next_power_of_two(int n)
{
    int res = 1;
    while (res < n)
        res <<= 1; // This code shifts the bit to the left until the
                   // power of two is greater than the input number.
    return res;
}

/**
 * @brief This function is used to calculate the value of the call option
 * based on the required strike price, array of strikes, and call option outputs.
 * (Completely generated using ChatGPT)
 *
 * @param K: The required strike price
 * @param strikes: The vector of strikes
 * @param call_prices: The vector of call options
 */
double linear_interpolate(double K,
                          const std::vector<double> &strikes,
                          const std::vector<double> &prices)
{
    // Bounds check
    if (K <= strikes.front())
        return prices.front();
    if (K >= strikes.back())
        return prices.back();

    // Binary search for index such that: strikes[i] <= K < strikes[i+1]
    auto upper = std::upper_bound(strikes.begin(), strikes.end(), K);
    size_t idx = std::distance(strikes.begin(), upper) - 1;

    double x0 = strikes[idx];
    double x1 = strikes[idx + 1];
    double y0 = prices[idx];
    double y1 = prices[idx + 1];

    // Linear interpolation formula
    double weight = (K - x0) / (x1 - x0);
    return y0 + weight * (y1 - y0);
}
/**
 * @brief Computes the price of a European call or put option using the Black-Scholes formula.
 *
 * This function implements the closed-form Black-Scholes model under risk-neutral assumptions.
 * It calculates the fair price of a European-style call or put option on a non-dividend-paying asset.
 *
 * @param S       Spot price of the underlying asset (S₀)
 * @param K       Strike price of the option
 * @param sigma   Volatility of the underlying asset (annualized)
 * @param r       Risk-free interest rate (annualized, continuously compounded)
 * @param T       Time to maturity (in years)
 * @param isCall  If true, returns the price of a call option; otherwise, returns the price of a put
 *
 * @return The Black-Scholes price of the option
 */

double BlackScholesPrice(double S,
                         double K,
                         double sigma,
                         double r,
                         double T,
                         bool isCall = true)
{
    // Guard against zero maturity or volatility
    if (T <= 0 || sigma <= 0)
    {
        if (isCall)
            return std::max(0.0, S - K);
        else
            return std::max(0.0, K - S);
    }

    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    auto N = [](double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2.0)); // CDF of standard normal
    };

    if (isCall)
        return S * N(d1) - K * std::exp(-r * T) * N(d2);
    else
        return K * std::exp(-r * T) * N(-d2) - S * N(-d1);
}

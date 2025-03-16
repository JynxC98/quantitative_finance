/**
 * @file explicit_scheme.cpp
 * @brief A script to simulate explicit finite difference scheme for BSM differential
 *        equation using cpp.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

using namespace std;

/**
 * @brief Calculate the cumulative distribution function (CDF) of a standard normal distribution
 * @param x: The input value for which the CDF is calculated.
 *
 * @returns The value of the standard normal CDF at x.
 */
double normcdf(double x)
{
    return 0.5 * erfc(-x * M_SQRT1_2); // M_SQRT1_2 is 1/sqrt(2)
}

/**
 * @brief Calculate the option price using the Black-Scholes model
 * @param spot: Current spot price.
 * @param strike: The predetermined strike price
 * @param T: The time to maturity.
 * @param r: The risk-free rate.
 * @param sigma: The current volatility.
 * @param isCall(optional): 1 if call option, else 0 for put option.
 *
 * @returns The option value.
 */
double calculateOptionPrice(double spot, double strike, double T, double r, double sigma, bool isCall = true)
{
    double d1 = (log(spot / strike) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    double option_value;

    option_value = isCall ? (spot * normcdf(d1) - strike * exp(-r * T) * normcdf(d2)) : (strike * exp(-r * T) * normcdf(-d2) - spot * normcdf(-d1));

    return option_value;
}

/**
 * @brief This function creates an equidistant grid between two ranges.
 * @param N: Number of grid points.
 * @param scaler_min: The starting point of the grid.
 * @param scaler_max: The end point of the grid.
 *
 * @returns A vector containing the equidistant grid points.
 */
vector<double> getEquidistantGrid(int N, double scaler_min, double scaler_max)
{
    // Evaluating the base condition.
    if (N < 2)
    {
        throw invalid_argument("There should be at least 2 values to create a grid");
    }

    // Initialising the grid with N data points.
    vector<double> result(N);

    // Calculating the range of the data points.
    double difference = scaler_max - scaler_min;

    // Calculating the grid value for the data points.
    double grid_value = difference / (N - 1);

    // Initialising the first value of the data grid.
    result[0] = scaler_min;

    for (int i = 1; i < N; ++i)
    {
        result[i] = result[i - 1] + grid_value;
    }
    return result;
}

/**
 * @brief This function simulates the explicit scheme for the Black-Scholes-Merton (BSM) differential equation
 *        using boundary conditions.
 * @param spot: Current spot price.
 * @param strike: The predetermined strike price
 * @param T: The time to maturity.
 * @param r: The risk-free rate.
 * @param sigma: The current volatility.
 * @param N(optional): Number of time steps in the grid.
 * @param M(optional): Number of price steps in the grid.
 * @param isCall(optional): 1 if call option, else 0 for put option.
 *
 * @returns A 2D vector containing the option price at each grid point.
 */
vector<vector<double>> explicitScheme(double spot,
                                      double strike,
                                      double T,
                                      double r,
                                      double sigma,
                                      int N = 100,
                                      int M = 100,
                                      bool isCall = true)
{
    // Setting the max spot price for the grid.
    double max_spot = 2.0 * spot; // Extend the spot range to capture sufficient values.

    // Initialising the time grid.
    auto time_grid = getEquidistantGrid(N, 0, T); // Assuming t = 0 to T.

    // Initialising the spot grid.
    auto spot_grid = getEquidistantGrid(M, 0, max_spot);

    // Calculating the step sizes.
    double dt = T / (N - 1);        // Time step size.
    double dS = max_spot / (M - 1); // Spot price step size.

    // This matrix will store the values of the option prices at various grid points.
    vector<vector<double>> option_data(N, vector<double>(M, 0.0));

    // Initialising the terminal condition for option_data.
    for (int j = 0; j < M; ++j)
    {
        // Terminal condition at t = T.
        option_data[N - 1][j] = isCall ? max(spot_grid[j] - strike, 0.0) : max(strike - spot_grid[j], 0.0);
    }

    // Initialising the boundary conditions.
    for (int n = 0; n < N; ++n)
    {
        // Boundary at S = 0.
        option_data[n][0] = isCall ? 0.0 : strike * exp(-r * time_grid[n]);

        // Boundary at S = max_spot.
        option_data[n][M - 1] = isCall ? (spot_grid[M - 1] - strike * exp(-r * time_grid[n])) : 0.0;
    }

    // Iterating backward in time.
    for (int n = N - 2; n >= 0; --n)
    {
        for (int i = 1; i < M - 1; ++i) // Avoiding boundary conditions at i = 0 and i = M-1.
        {
            double a = 0.5 * dt * (pow(sigma, 2) * pow(i, 2) - r * i);
            double b = 1 - dt * (pow(sigma, 2) * pow(i, 2) + r);
            double c = 0.5 * dt * (pow(sigma, 2) * pow(i, 2) + r * i);

            // Upgrading grid using the explicit scheme formula.
            option_data[n][i] = a * option_data[n + 1][i - 1] + b * option_data[n + 1][i] + c * option_data[n + 1][i + 1];
        }
    }

    return option_data;
}

/**
 * @brief The main function to execute the explicit finite difference scheme.
 */
int main()
{
    // Defining parameters for the option.
    double spot = 100.0;
    double strike = 110.0;
    double T = 1.0;
    double r = 0.04;
    double sigma = 0.25;

    // Getting the option prices using the explicit scheme.
    auto option_data = explicitScheme(spot, strike, T, r, sigma);

    // Printing the option prices at t = 0 for all spot prices.
    for (const auto &element : option_data[0])
    {
        cout << element << " ";
    }
    cout << endl;

    return 0;
}

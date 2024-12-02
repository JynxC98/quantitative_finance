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

using namespace std;

/**
 * @brief This function creates an equidistant grid between two ranges.
 * @param N: Number of grid points.
 * @param scaler_min: The starting point of the grid.
 * @param scaler_max: The end point of the grid.
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

    double grid_value = double(difference / (N - 1));

    // Initialising the first value of the data grid .

    result[0] = scaler_min;

    for (int i = 1; i < N; ++i)
    {
        result[i] = result[i - 1] + grid_value;
    }
    return result;
}

/**
 * @brief This function simulates the explicit scheme for BSM differential equation
 *        using the boundary level conditions.
 * @param spot: Current spot price.
 * @param strike: The predetermined strike price
 * @param T: The time to maturity.
 * @param r: The risk free fate.
 * @param sigma: The current volatiity.
 * @param N(optional): Time grid.
 * @param M(optional): Underlying grid
 * @param isCall(optional): 1 if call else 0 for put.
 *
 * @returns: vector<double> option price at t = 0.
 */
double explicitScheme(double spot,

                      double strike,
                      double T,
                      double r,
                      double sigma,
                      int N = 100,
                      int M = 100,
                      bool isCall = true)
{

    // Setting the max spot price data.

    int multiplier = 2;
    double max_spot = double(multiplier) * spot;

    // Initialising the time grid.

    auto time_grid = getEquidistantGrid(N, 0, T); // Assuming t=0

    // Intialising the spot grid.

    auto spot_grid = getEquidistantGrid(M, spot, max_spot);

    // This matrix will store the values of the option prices at various grid points.

    vector<vector<double>> option_data(M, vector<double>(N, 0.0));

    // Initialising the boundary condition for the option_data

    // Setting up terminal conditions
    for (int j = 0; j < M; ++j)
    {
        // Terminal condition at t = T
        option_data[j][N - 1] = isCall ? max(spot_grid[j] - strike, 0.0) : max(strike - spot_grid[j], 0.0);
    }

    // Setting up boundary conditions

    for (int i = 0; i < N; ++i)
    {
        // Boundary at S = 0
        option_data[0][i] = isCall ? 0.0 : strike * exp(-r * time_grid[i]);

        // Boundary at S = max_spot
        option_data[M - 1][i] = isCall ? max_spot - strike * exp(-r * time_grid[i]) : 0.0;
    }

    // Iterating backwards in time
    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = 0; j < M; ++j)
        {
            {};
        }
    }
}

int main()
{

    double spot = 100.0;
    double spot_max = 200;
    int N = 4;
    auto result = getEquidistantGrid(N, spot, spot_max);

    for (const auto &element : result)
    {
        cout << element << " ";
    }
    return 0;
}
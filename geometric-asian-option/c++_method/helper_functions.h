/**
 * @file helper_functions.h
 * @brief This file stores a set of helper functions for the `heston_analytical_solution` file.
 *
 * @author Harsh Parikh
 * @date 12th April 2025
 */

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

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
 * @brief This function returns a vector of numbers from a particular range
 *
 * @param min_element: The lower limit of the range
 * @param max_element: The upper limit of the range
 */
vector<int> getArange(int min_element, int max_element)
{

    // If the min element is greater than the max element, an empty array is return
    if (min_element >= max_element)
        return {};

    // Calculating the number of elements to be stored in the container
    int num_elements = max_element - min_element;

    // This vector stores the required results
    vector<int> arange(num_elements);

    for (int i = 0; i < num_elements; ++i)
    {
        arange[i] = min_element + i;
    }

    return arange;
}
/**
 * @brief This function calculates the integral of a function using the trapezoidal
 * method.
 *
 * @param upper_limit: The upper limit of the integral
 * @param lower_limit: The lower limit of the integral
 * @param integrand: The function to be integrated
 * @param N: The number of grid points
 */
template <typename function>
double getIntegralTrapezoidal(function integrand,
                              double lower_limit,
                              double upper_limit,
                              int N)
{
    // Calculating the value of \delta x
    double dx = (upper_limit - lower_limit) / N;

    // Fetching the equidistant grid
    auto grid_points = getEquidistantGrid(N + 1, lower_limit, upper_limit);

    // Evaluating the value of the integral using the trapezoidal rule

    double average_range = 0.5 * (integrand(grid_points[0]) +
                                  integrand(grid_points[N])); // Calculating the average of the integrand
                                                              // at the range points

    // This variable stores the floating sum of the integrand values
    // from range 1 to N - 1
    double floating_sum = 0.0;
    for (int i = 1; i < N - 1; ++i)
    {
        floating_sum += integrand(grid_points[i]);
    }

    // Calculating the value of the integral based on floating sum
    auto required_value = dx * (average_range + floating_sum);

    return required_value;
}
#endif

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
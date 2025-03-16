#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief This function returns the equidistant vector between two provided values
 * @param lower_value: The lower limit of the function
 * @param upper_value: The upper limit of the function
 * @param num_elements: The length of the grid
 */
template <typename data_type>
vector<double> getEquidistantGrid(data_type lower_value,
                                  data_type upper_value,
                                  int num_elements)

{

    // Calculating the range of the values
    auto range = upper_value - lower_value;

    // Calculating the spatial discretisation
    double dx = range / (num_elements - 1);

    // This vector stores the equidistant grid
    vector<double> grid(num_elements, 0.0);

    // Initialising the first element of the grid with the lower value
    grid[0] = lower_value;

    for (int i = 1; i < num_elements; ++i)
    {
        grid[i] = grid[i - 1] + dx;
    }
    return grid;
}

#endif
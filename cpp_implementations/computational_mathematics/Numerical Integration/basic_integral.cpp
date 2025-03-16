/**
 * @file basic_integral.cpp
 *
 * @brief A basic Monte-Carlo method of evaluating an integral
 * @author Harsh Parikh
 * @date 16-03-2025
 */

#include <iostream>
#include "helper_functions.h"
#include <cmath>
#include <vector>

using namespace std;

/**
 * @brief A basic integral function
 */

double integralFuntion(double x)
{
    return sin(x);
}

/**
 * @brief This function evaluates the integral.
 *
 * @param integrand: The integral function to be evaluated.
 * @param lower_limit: The lower limit of the integral.
 * @param upper_limit: The upper limit of the integral.
 */
template <typename Func>
double evaluateIntegral(Func integrand,
                        double lower_limit,
                        double upper_limit,
                        int N = 5000)
{
    // Calculating the spatial gridpoint
    double dx = (upper_limit - lower_limit) / N;

    // This vector will store the discretised grid
    auto grid = getEquidistantGrid(lower_limit, upper_limit, N);

    // Storing the sum of integral values
    double result = 0.0;

    for (const auto &element : grid)
    {
        result += integrand(element) * dx;
    }
    return result;
}

int main()
{

    // Evaluating the integral of sin(x) from zero to pi/2, the result should
    // be close to 1.

    double lower_limit = 0.0;
    double upper_limit = M_PI / 2;
    int N = 5000;

    auto result = evaluateIntegral(integralFuntion, lower_limit, upper_limit, N);

    cout << "The value of the required integral is " << result << endl;

    return 0;
}

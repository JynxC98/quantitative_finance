/**
 * @brief This function is used to calculate the value of Phi inverse using
 * Newton's root finding algorithm. The key idea is to converge the
 * values until the tolerance is achieved or maximum iterations
 * are covered.
 *
 * @author Harsh Parikh
 * @date 18th April 2026
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @brief This function calculates the CDF of the standard normal distribution.
 *
 * @param x: The variable
 * @returns: The area under the curve for the presented value.
 */
template <typename data_type>
long double CDF(data_type x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

/**
 * @brief This function calculates the PDF of the standard normal distribution.
 * @param x: The variable
 *
 * @returns: The probability of the presented value.
 */
template <typename data_type>
long double PDF(data_type x)
{

    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-(x * x) / 2.0);
}

/**
 * @brief This function calculates the phi inverse for the standard normal distribution.
 *
 * @param x: The variable
 * @returns: The value associated to the area under the curve.
 */
template <typename data_type>
long double phi_inverse(data_type x)
{

    if ((x <= 0) || (x >= 1))
    {
        throw std::invalid_argument("Please enter the value between 0 and 1, excluding them");
    }

    double tolerance = 1e-12;
    int max_iterations = 1000;

    long double previous = 0.0; // Initial guess for our value

    long double current;

    for (int i = 0; i < max_iterations; ++i)
    {
        current = previous - ((CDF(previous) - x) / (PDF(previous)));

        if (std::abs(current - previous) < tolerance)
        {
            return current;
        }
        previous = current;
    }

    std::cout << "The Newton's method failed to converge";

    return std::numeric_limits<long double>::quiet_NaN();
}

int main()
{

    // Running some validation checks.
    double x = 0.975; // Phi inverse of 0.975 should return 1.96.

    auto val = phi_inverse(x);
    std::cout << val << std::endl;

    return 0;
}
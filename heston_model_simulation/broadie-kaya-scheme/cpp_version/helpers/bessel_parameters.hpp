/**
 * @brief This header file is used to store the Bessel parameters.
 *
 * @author Harsh Parikh
 */

#if !defined(BESSEL_PARAMETERS_HPP)
#define BESSEL_PARAMETERS_HPP

#include <iostream>

struct BesselParams
{
    int num_iterations = 100;
    double tolerance = 1e-8;
    double threshold = 10.0;
    bool log_space = true;
};

#endif
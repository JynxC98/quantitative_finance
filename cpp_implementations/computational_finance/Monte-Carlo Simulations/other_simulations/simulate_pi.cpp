/**
 * @file simulate_pi.cpp
 * @brief A script that calculates the value of pi using Monte-Carlo simulations
 * @author Harsh Parikh
 */

#include <iostream>
#include <random>

using namespace std;

double simulatePI(int num_iterations)
{
    double PI = 0.0;
    int inside_circle = 0;

    // Create a random device and Mersenne Twister generator
    random_device rd;
    mt19937 gen(rd());

    // Define a uniform distribution between 0 and 1
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int itr = 0; itr < num_iterations; ++itr)
    {
        double x = dis(gen); // Random x-coordinate
        double y = dis(gen); // Random y-coordinate

        // Check if the point lies inside the unit circle
        if (x * x + y * y <= 1.0)
        {
            inside_circle++;
        }
    }

    // Calculate the approximation of PI
    PI = 4.0 * static_cast<double>(inside_circle) / num_iterations;
    return PI;
}

int main()
{
    // Number of iterations for the Monte Carlo simulation
    int num_iterations = 1000000; // 1 million

    // Simulate PI and print the result
    double pi_estimate = simulatePI(num_iterations);
    cout << "Estimated value of PI: " << pi_estimate << endl;

    return 0;
}

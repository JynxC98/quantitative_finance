/**
 * @file european_model.cpp
 * @brief A script to calculate the European option price using binomial tree method.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <cmath> // Include cmath for exp and pow

using namespace std;

double getEuropeanPrice(double spot,
                        double strike,
                        double sigma,
                        double r,
                        double T,
                        bool isCall,
                        int N = 100)
{
    // Calculating the time step
    double dt = T / N;

    // Calculating the up and down factors
    double u = exp(sigma * sqrt(dt));
    double d = 1.0 / u;

    // Calculating the risk-neutral probabilities
    double p = (exp(r * dt) - d) / (u - d);
    double q = 1 - p;

    // Creating the option price grid
    vector<double> option_grid(N + 1);

    // Initializing the terminal values for the option payoffs
    for (int i = 0; i <= N; ++i)
    {
        double current_spot = spot * pow(u, N - i) * pow(d, i);
        option_grid[i] = isCall ? max(current_spot - strike, 0.0) : max(strike - current_spot, 0.0);
    }

    // Traversing backward
    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = 0; j <= i; ++j)
        {
            option_grid[j] = exp(-r * dt) * (p * option_grid[j] + q * option_grid[j + 1]);
        }
    }

    return option_grid[0];
}

int main()
{
    double spot = 100.0;
    double strike = 110.0;
    double sigma = 0.25;
    double T = 1.0;
    double r = 0.045;
    bool isCall = true;

    double price = getEuropeanPrice(spot, strike, sigma, r, T, isCall);

    cout << "The price of the option is " << price << endl;
    return 0;
}

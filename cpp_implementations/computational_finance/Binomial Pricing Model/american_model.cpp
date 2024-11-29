/**
 * @file american_model.cpp
 * @brief A script to calculate the American option price using binomial tree method.
 * @author Harsh Parikh
 */
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double getAmericanOptionPrice(double spot,
                              double strike,
                              double sigma,
                              double r,
                              double T,
                              bool isCall,
                              int N = 100)
{
    // Time step
    double dt = T / N;

    // Up factor
    double u = exp(sigma * sqrt(dt));

    // Down factor
    double d = 1.0 / u;

    // Risk-neutral probability
    double p = (exp(r * dt) - d) / (u - d);
    double q = 1 - p;

    // Option price grid
    vector<double> option_grid(N + 1);

    // Initialising the terminal payoff
    for (int i = 0; i <= N; ++i)
    {
        double current_spot = spot * pow(u, N - i) * pow(d, i);
        option_grid[i] = isCall ? max(current_spot - strike, 0.0) : max(strike - current_spot, 0.0);
    }

    // Iterating backwards in time
    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = 0; j <= i; ++j)
        {
            // Spot price at current node
            double current_spot = spot * pow(u, i - j) * pow(d, j);

            // Present value of holding the option
            double present_value = exp(-r * dt) * (p * option_grid[j] + q * option_grid[j + 1]);

            // Intrinsic value of exercising the option early
            double intrinsic_value = isCall ? max(current_spot - strike, 0.0) : max(strike - current_spot, 0.0);

            // American option value is the maximum of holding or exercising
            option_grid[j] = max(present_value, intrinsic_value);
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

    double price = getAmericanOptionPrice(spot, strike, sigma, r, T, isCall);

    cout << "The price of the option is " << price << endl;
    return 0;
}

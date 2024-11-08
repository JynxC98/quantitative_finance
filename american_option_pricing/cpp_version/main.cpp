#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

double american_option(double spot, double strike, double T, double rate, double volatility, int steps, bool isCall)
{
    // Time per step
    double dt = T / steps;

    // Discount factor per step
    double discount = exp(-rate * dt);

    // Up and down factors based on volatility and time per step
    double u = exp(volatility * sqrt(dt));
    double d = 1.0 / u;

    // Risk neutral probability
    double p = (exp(rate * dt) - d) / (u - d);

    // Vector to store option values at each node in the last layer
    vector<double> option_values(steps + 1);

    // Final payoff at maturity for all nodes
    for (int i = 0; i <= steps; i++)
    {
        double ST = spot * pow(u, i) * pow(d, steps - i); // Stock price at node (steps, i)
        if (isCall)
        {
            option_values[i] = max(0.0, ST - strike); // Call payoff
        }
        else
        {
            option_values[i] = max(0.0, strike - ST); // Put payoff
        }
    }

    // Backward induction through the tree
    for (int step = steps - 1; step >= 0; step--)
    {
        for (int i = 0; i <= step; i++)
        {
            double ST = spot * pow(u, i) * pow(d, step - i); // Stock price at node (step, i)

            // Calculate option value if held to the next step
            option_values[i] = discount * (p * option_values[i + 1] + (1 - p) * option_values[i]);

            // For American options, check if early exercise is optimal
            if (isCall)
            {
                option_values[i] = max(option_values[i], ST - strike); // Call
            }
            else
            {
                option_values[i] = max(option_values[i], strike - ST); // Put
            }
        }
    }

    // The option price is now at the root of the tree
    return option_values[0];
}

int main()
{
    // Define option parameters
    double spot = 100.0;     // Current stock price
    double strike = 110.0;   // Strike price
    double T = 1.0;          // Time to maturity in years
    double rate = 0.04;      // Risk-free interest rate
    double volatility = 0.2; // Volatility of the stock
    int steps = 1000;        // Number of steps in the binomial tree
    bool isCall = true;      // true for call option, false for put option

    // Calculate option price
    double option_price = american_option(spot, strike, T, rate, volatility, steps, isCall);

    // Output the result
    cout << "American option price: " << option_price << endl;

    return 0;
}

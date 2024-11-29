/**
 * @file: options.cpp
 * @brief: A script to calculate option prices and the corresponding greeks.
 * @author: Harsh Parikh
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <armadillo>

using namespace std;
using namespace arma;

/**
 * @class OptionProperties
 * @brief Handles option pricing, Greeks, and volatility calculations
 */
class OptionProperties
{
private:
    double spot;      // Current spot price
    double strike;    // Strike price
    double T;         // Time to maturity
    double sigma;     // Volatility
    double r;         // Risk free rate
    bool option_type; // Option type, true for call, false for put

public:
    /**
     * @brief Constructor for OptionProperties
     * @param spot Current spot price
     * @param strike Strike price
     * @param T Time to maturity
     * @param sigma Volatility
     * @param r Risk-free rate
     * @param option_type Option type (true for call, false for put)
     */
    OptionProperties(double spot, double strike, double T, double sigma, double r, bool option_type)
        : spot(spot), strike(strike), T(T), sigma(sigma), r(r), option_type(option_type) {}

    // Declare member functions
    double getOptionPrice(double sigma);
    map<string, double> getGreeks();
    double calculateImpVol(double market_price);
};

/**
 * @brief Calculate option price using Black-Scholes model
 * @param sigma Volatility
 * @return Option price
 */
double OptionProperties::getOptionPrice(double sigma)
{
    double d1 = (log(spot / strike) + (r + 0.5 * (sigma * sigma)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    if (option_type) // Call option
    {
        return spot * normcdf(d1) - strike * exp(-r * T) * normcdf(d2);
    }
    else // Put option
    {
        return strike * exp(-r * T) * normcdf(-d2) - spot * normcdf(-d1);
    }
}

/**
 * @brief Calculate option Greeks
 * @return Map of Greek values
 */
map<string, double> OptionProperties::getGreeks()
{
    map<string, double> greeks;
    double d1 = (log(spot / strike) + (r + 0.5 * (sigma * sigma)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    // Gamma (same for both call and put)
    greeks["Gamma"] = normpdf(d1) / (spot * sigma * sqrt(T));

    // Calculate greeks based on option type
    // Call option
    if (option_type)
    {
        // Delta
        greeks["Delta"] = normcdf(d1);

        // Theta (corrected)
        double theta_term = -(spot * sigma * normpdf(d1)) / (2 * sqrt(T));
        greeks["Theta"] = theta_term - r * strike * exp(-r * T) * normcdf(d2);

        // Vega
        greeks["Vega"] = spot * sqrt(T) * normpdf(d1);

        // Rho
        greeks["Rho"] = strike * T * exp(-r * T) * normcdf(d2);
    }
    // Put option
    else
    {
        // Delta
        greeks["Delta"] = normcdf(d1) - 1;

        // Theta (corrected)
        double theta_term = -(spot * sigma * normpdf(d1)) / (2 * sqrt(T));
        greeks["Theta"] = theta_term + r * strike * exp(-r * T) * normcdf(-d2);

        // Vega
        greeks["Vega"] = spot * sqrt(T) * normpdf(d1);

        // Rho
        greeks["Rho"] = -strike * T * exp(-r * T) * normcdf(-d2);
    }
    return greeks;
}

/**
 * @brief Calculate implied volatility using Newton-Raphson method
 * @param market_price Market price of the option
 * @return Implied volatility
 */
double OptionProperties::calculateImpVol(double market_price)
{
    /*
    Here,
    f(x) = model_price - market_price

    volatility_{t+1} = volatility_t - (model_price - market_price) / vega
    */
    double tolerance = 1e-6;
    double vol_low = 0.0001;
    double vol_high = 5;
    double current_vol = (vol_low + vol_high) / 2; // Initial guess

    for (int i = 0; i < 100; ++i)
    {
        sigma = current_vol;
        double model_price = getOptionPrice(current_vol);

        // Calculate vega
        double vega = getGreeks()["Vega"];

        // Newton-Raphson method
        double diff = model_price - market_price;

        if (abs(diff) < tolerance)
        {
            return current_vol;
        }

        current_vol -= diff / vega;
    }

    return current_vol; // Best approximation
}

/**
 * @brief Main function to demonstrate option pricing
 * @return Exit status
 */
int main()
{
    // Example option parameters
    double spot = 100.0;
    double strike = 110.0;
    double T = 1.0;
    double sigma = 0.25;
    double r = 0.045;
    bool option_type = true; // Call option

    // Create option object
    OptionProperties option(spot, strike, T, sigma, r, option_type);

    // Print option price
    cout << fixed << setprecision(4);
    cout << "Option Price: " << option.getOptionPrice(sigma) << endl;

    // Calculate and print Greeks
    auto greeks = option.getGreeks();
    cout << "\nOption Greeks:" << endl;
    for (const auto &greek : greeks)
    {
        cout << greek.first << ": " << greek.second << endl;
    }

    // Demonstrate implied volatility calculation
    double market_price = 10.0;
    double implied_vol = option.calculateImpVol(market_price);
    cout << "\nImplied Volatility: " << implied_vol << endl;

    return 0;
}
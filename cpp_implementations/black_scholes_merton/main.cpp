#include <iostream>
#include <cmath>

/**
 * @brief Black-Scholes Option Pricing Model
 * 
 * This class implements the Black-Scholes model for European option pricing.
 * It provides methods to calculate call and put option prices, as well as
 * various Greeks (delta, gamma, vega, theta, and rho).
 */
class BlackScholes {
private:
    double S; // Current stock price
    double K; // Strike price
    double r; // Risk-free interest rate
    double T; // Time to expiration (in years)
    double sigma; // Volatility

    /**
     * @brief Calculate the d1 parameter for the Black-Scholes formula
     * @return double The d1 parameter
     */
    double calculateD1() const {
        return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    }

    /**
     * @brief Calculate the d2 parameter for the Black-Scholes formula
     * @return double The d2 parameter
     */
    double calculateD2() const {
        return calculateD1() - sigma * sqrt(T);
    }

    /**
     * @brief Calculate the cumulative normal distribution
     * @param x The input value
     * @return double The cumulative normal distribution value
     */
    double normalCDF(double x) const {
        return 0.5 * (1 + erf(x / sqrt(2)));
    }

public:
    /**
     * @brief Construct a new Black Scholes object
     * @param S Current stock price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param T Time to expiration (in years)
     * @param sigma Volatility
     */
    BlackScholes(double S, double K, double r, double T, double sigma)
        : S(S), K(K), r(r), T(T), sigma(sigma) {}

    /**
     * @brief Calculate the price of a European call option
     * @return double The call option price
     */
    double callPrice() const {
        double d1 = calculateD1();
        double d2 = calculateD2();
        return S * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
    }

    /**
     * @brief Calculate the price of a European put option
     * @return double The put option price
     */
    double putPrice() const {
        double d1 = calculateD1();
        double d2 = calculateD2();
        return K * exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
    }

    /**
     * @brief Calculate the delta of the option
     * @param isCall True for call option, false for put option
     * @return double The delta value
     */
    double delta(bool isCall) const {
        double d1 = calculateD1();
        return isCall ? normalCDF(d1) : normalCDF(d1) - 1;
    }

    /**
     * @brief Calculate the gamma of the option
     * @return double The gamma value
     */
    double gamma() const {
        double d1 = calculateD1();
        return exp(-d1 * d1 / 2) / (S * sigma * sqrt(2 * M_PI * T));
    }

    /**
     * @brief Calculate the vega of the option
     * @return double The vega value
     */
    double vega() const {
        double d1 = calculateD1();
        return S * sqrt(T) * exp(-d1 * d1 / 2) / sqrt(2 * M_PI);
    }

    /**
     * @brief Calculate the theta of the option
     * @param isCall True for call option, false for put option
     * @return double The theta value
     */
    double theta(bool isCall) const {
        double d1 = calculateD1();
        double d2 = calculateD2();
        double term1 = -S * sigma * exp(-d1 * d1 / 2) / (2 * sqrt(2 * M_PI * T));
        double term2 = r * K * exp(-r * T) * normalCDF(isCall ? d2 : -d2);
        return isCall ? term1 - term2 : term1 + term2;
    }

    /**
     * @brief Calculate the rho of the option
     * @param isCall True for call option, false for put option
     * @return double The rho value
     */
    double rho(bool isCall) const {
        double d2 = calculateD2();
        return isCall ? K * T * exp(-r * T) * normalCDF(d2) : -K * T * exp(-r * T) * normalCDF(-d2);
    }
};

int main() {
    // Example usage
    double S = 100;  // Current stock price
    double K = 100;  // Strike price
    double r = 0.05; // Risk-free rate (5%)
    double T = 1;    // Time to expiration (1 year)
    double sigma = 0.2; // Volatility (20%)

    BlackScholes bs(S, K, r, T, sigma);

    std::cout << "Call Price: " << bs.callPrice() << std::endl;
    std::cout << "Put Price: " << bs.putPrice() << std::endl;
    std::cout << "Call Delta: " << bs.delta(true) << std::endl;
    std::cout << "Put Delta: " << bs.delta(false) << std::endl;
    std::cout << "Gamma: " << bs.gamma() << std::endl;
    std::cout << "Vega: " << bs.vega() << std::endl;
    std::cout << "Call Theta: " << bs.theta(true) << std::endl;
    std::cout << "Put Theta: " << bs.theta(false) << std::endl;
    std::cout << "Call Rho: " << bs.rho(true) << std::endl;
    std::cout << "Put Rho: " << bs.rho(false) << std::endl;

    return 0;
}
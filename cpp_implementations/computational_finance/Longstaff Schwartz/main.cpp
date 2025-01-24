/**
 * @file main.cpp
 * @brief A script to calculate the value of an American call option using the
 * Longstaff Schwartz method.

 * @date 20-December-2024
 * @author Harsh Parikh
 */

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <algorithm>

using namespace std;

/**
 * @brief An implementation of the Longstaff Schwartz Model (LSM) to price American options
          efficiently. The LSM model uses dynamic programming to find the optimal stopping
          point and uses Monte-Carlo simulations to calculate the expected value of the option.

 * @cite 1. https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf
 * @cite 2. https://uu.diva-portal.org/smash/get/diva2:818128/FULLTEXT01.pdf
 *
 * @param spot: The current spot price
 * @param strike: The predetermined strike price
 * @param r: The risk-free rate
 * @param sigma: The underlying volatility
 * @param T: Time to maturity
 * @param M (optional, default = 5000): The number of Monte-Carlo paths
 * @param N (optional, default = 5000): The number of time steps
 * @param isCall(optional, default = true): To indicate wheather the option is a call (1) or put (0)
 */

class LongstaffShwartzAlgorithm
{
private:
    double spot;
    double strike;
    double r;
    double sigma;
    double T;
    int M;
    int N;
    bool isCall;

public:
    LongstaffShwartzAlgorithm(double spot,
                              double strike,
                              double r,
                              double sigma,
                              double T,
                              int M = 5000,
                              int N = 5000,
                              bool isCall = true) : spot(spot), strike(strike), T(T), r(r), sigma(sigma), M(M), N(N), isCall(isCall)
    {
    }

    /**
     * @brief This function calculates the value of coefficients using the Ordinary Least Squares (OLS) method.
     * OLS is given as:
     *
     * \beta = (X^T . X) ^ -1 . X^T .y
     * Where,
     *
     * \beta: Coefficient vector
     *
     * X: Feature matrix
     *
     * y: Target matrix
     *
     * @param feature_matrix: The input feature matrix of size m(num features) x n(num target)
     * @param target_matrix: The target matrix of size 1 x n (num target)
     *
     * @returns
     * Coefficient matrix of size 1 x n
     */
    Eigen::MatrixXd OrdinaryLeastSquares(const Eigen::MatrixXd &feature_MatrixXdrix, const Eigen::MatrixXd &target_vector);

    Eigen::MatrixXd getBrownianPaths();
};

Eigen::MatrixXd LongstaffShwartzAlgorithm ::OrdinaryLeastSquares(const Eigen::MatrixXd &feature_MatrixXdrix,
                                                                 const Eigen::MatrixXd &target_vector)
{
    return {};
}
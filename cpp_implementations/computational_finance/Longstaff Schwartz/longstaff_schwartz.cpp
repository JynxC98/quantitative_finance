/**
 * @file longstaff_schwartz.cpp
 * @brief A script to calculate the value of an American call option using the
 * Longstaff Schwartz method.

 * @date 20-December-2024
 * @author Harsh Parikh
 */

#include <iostream>
#include <armadillo>
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
    arma::mat OrdinaryLeastSquares(const arma::mat &feature_matrix, const arma::mat &target_vector);

    arma ::mat getBrownianPaths();
};

arma::mat LongstaffShwartzAlgorithm ::OrdinaryLeastSquares(const arma::mat &feature_matrix,
                                                           const arma::mat &target_vector)
{

    // Ensuring the dimensions are valid
    if (feature_matrix.n_rows != target_vector.n_rows)
    {
        throw std::invalid_argument("Number of rows in feature_matrix and target_vector must match.");
    }

    // Calculating the transpose of the feature matrix
    arma::mat X_transpose = feature_matrix.t();

    // Calculating (X^T * X)
    arma::mat XtX = X_transpose * feature_matrix;

    // Calculating the inverse of (X^T * X)
    arma::mat XtX_inv = arma::inv(XtX);

    // Calculating (X^T * y)
    arma::mat Xty = X_transpose * target_vector;

    // Final coefficients: beta = (X^T * X)^-1 * (X^T * y)
    arma::mat beta = XtX_inv * Xty;

    return beta;
}

int main()
{
    // Initialising the option data
    double spot = 100;
    double strike = 110;
    double sigma = 0.25;
    double r = 0.045;
    double T = 1.0;

    // Initialising the LongstaffSchwartz class
    LongstaffShwartzAlgorithm instance(spot, strike, r, sigma, T);
    // Checking the OLS function
    arma::mat feature_matrix = {{1, 2, 4}, {3, 5, 6}, {4, 2, 1}};
    arma::mat target_vector = {1, -4, 3};

    // Initialising the LongstaffSchwartz constructor

    // Calculating the coefficients using the OLS
    return 0;
}
/**
 * @file linear_regression
 * @brief Implements ordinary least squares regression on input data vectors.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense> // For all matrix computations.

using namespace std;

/**
 * @brief This class performs the Ordinary Least Squares (OLS) regression on the input
 * vectors. The formula for OLS is given by
 * \beta = ((X^T*X)^-1)* X^T *y
 * where,
 * \beta: Coefficients
 * X: feature matrix
 * y: target matrix
 *
 * @param feature_matrix: The feature matrix.
 * @param target_vector: The target vector.
 *
 * @returns The coefficients for the OLS regression.
 */
template <typename T>
class OrdinaryLeastSquares
{
private:
    // The feature matrix (Dynamic number of rows and columns)
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &feature_matrix;

    // The target vector (Dynamic number of rows and 1 column)
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &target_vector;

public:
    /**
     * @brief Constructor to initialize the feature matrix and target vector.
     *
     * @param feature_matrix: The input feature matrix.
     * @param target_vector: The target vector.
     */
    OrdinaryLeastSquares(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &feature_matrix,
                         const Eigen::Matrix<T, Eigen::Dynamic, 1> &target_vector) : feature_matrix(feature_matrix),
                                                                                     target_vector(target_vector) {}

    /**
     * @brief This method is used to compute the OLS Coefficients.
     * @returns The vector of regression coefficients.
     */
    Eigen::Matrix<T, Eigen::Dynamic, 1> computeOLSCoefficients();
};

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> OrdinaryLeastSquares<T>::computeOLSCoefficients()
{
    if (feature_matrix.rows() != target_vector.rows())
    {
        throw std::invalid_argument("Feature matrix rows and target vector size must match.");
    }

    // Compute OLS coefficients using the formula: β = (X^T X)^-1 X^T y
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> XTX = feature_matrix.transpose() * feature_matrix;
    Eigen::Matrix<T, Eigen::Dynamic, 1> XTy = feature_matrix.transpose() * target_vector;

    // Solve for coefficients: β = (X^T X)^-1 X^T y
    Eigen::Matrix<T, Eigen::Dynamic, 1> coefficients = XTX.ldlt().solve(XTy);

    return coefficients;
}

int main()
{

    // Initialize the feature matrix (3 samples, 3 features)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> feature_matrix(3, 3);
    feature_matrix << 1, 2, 3,
        2, 3, 5,
        7, 7, 9;

    // Add a column of ones for the intercept term
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> augmented_matrix(feature_matrix.rows(), feature_matrix.cols() + 1);
    augmented_matrix << Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(feature_matrix.rows()), feature_matrix;

    // Initialize the target vector
    Eigen::Matrix<double, Eigen::Dynamic, 1> target_vector(3);
    target_vector << 1,
        4,
        5;

    // Create an instance of the OrdinaryLeastSquares class
    OrdinaryLeastSquares<double> ols(augmented_matrix, target_vector);

    // Compute the OLS coefficients
    Eigen::Matrix<double, Eigen::Dynamic, 1> coefficients = ols.computeOLSCoefficients();

    // Print the coefficients
    cout << "OLS Coefficients:\n";
    for (int i = 0; i < coefficients.size(); ++i)
    {
        cout << coefficients[i] << " ";
    }
    cout << endl;

    return 0;
}

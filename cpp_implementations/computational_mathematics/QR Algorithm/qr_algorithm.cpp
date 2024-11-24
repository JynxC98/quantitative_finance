/**
 * @file qr_algorithm.cpp
 * @brief A script that performs QR decomposition of a matrix.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>

using namespace std;

template <typename T>
class QRAlgorithm
{
    /**
     * @brief QR algorithm is an iterative method used to compute the eigenvalues
     * and eigenvectors of a square matrix by factorising the matrix into:
     *
     * A = Q * R,
     * where,
     * A: The original square matrix
     * Q: Orthogonal matrix
     * R: Upper triangular matrix
     *
     * This method applies the QR factorisation iteratively, updating A as:
     * A_k+1 = R_k * Q_k,
     * until convergence to a nearly diagonal matrix.
     *
     * The diagonal entries of the converged matrix represent the eigenvalues.
     *
     * @param matrix: Reference to the square matrix to process.
     * @param maxIterations: Maximum number of iterations for the algorithm.
     * @param tolerance: Convergence threshold for the off-diagonal entries.
     * @return A pair containing a matrix of eigenvalues and a matrix of eigenvectors if the process converges.
     */

private:
    const vector<vector<T>> &matrix; // Reference to the original input matrix
    int maxIterations;
    double tolerance;

public:
    // Constructor with member initializer list
    QRAlgorithm(const vector<vector<T>> &matrix,
                int maxIterations = 1000,
                double tolerance = 1e-10)
        : matrix(matrix), maxIterations(maxIterations), tolerance(tolerance) {}

    void displayMatrix(vector<vector<T>> &inputMatrix);
    T compute_norm(vector<T> _vector);
    vector<vector<T>> getTranspose(vector<vector<T>> &inputMatrix);
    vector<vector<T>> mutliplyMatrices(vector<vector<T>> inputMatrix_A, inputMatrix_B);
    T calculate_L2_norm(vector<vector<T>> _vector);
};

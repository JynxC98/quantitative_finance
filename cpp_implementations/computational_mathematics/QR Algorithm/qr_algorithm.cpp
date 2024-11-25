/**
 * @file qr_algorithm.cpp
 * @brief A script that performs QR decomposition of a matrix using Householder method.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>

using namespace std;

class QRDecomposition
{
    /**
     * @brief QR algorithm is an iterative method used to compute the eigenvalues
     * and eigenvectors of a square matrix by factorizing the matrix into:
     *
     * A = Q * R,
     * where,
     * A: The original square matrix
     * Q: Orthogonal matrix
     * R: Upper triangular matrix
     *
     * This method applies the QR factorization iteratively, updating A as:
     * A_k+1 = R_k * Q_k,
     * until convergence to a nearly diagonal matrix.
     *
     * The diagonal entries of the converged matrix represent the eigenvalues.
     *
     * @param matrix: Reference to the square matrix to process.
     * @param maxIterations: Maximum number of iterations for the algorithm.
     * @param tolerance: Convergence threshold for the off-diagonal entries.
     */
private:
    const vector<vector<double>> &matrix; // Reference to the original input matrix
    int maxIterations;
    double tolerance;

public:
    // Constructor with member initializer list
    QRDecomposition(const vector<vector<double>> &matrix,
                    int maxIterations = 1000,
                    double tolerance = 1e-10)
        : matrix(matrix), maxIterations(maxIterations), tolerance(tolerance) {}

    /**
     * @brief The function prints the input matrix with proper formatting.
     * @param matrix The matrix to be displayed
     */
    void displayMatrix(const vector<vector<double>> &matrix) const
    {
        for (const auto &row : matrix)
        {
            for (const auto &element : row)
            {
                cout << fixed << setprecision(6) << setw(12) << element << " ";
            }
            cout << endl;
        }
    }

    /**
     * @brief The function calculates the L2 norm of a vector.
     * @param vec Input vector
     * @return L2 norm of the vector
     */
    double calculateL2Norm(const vector<double> &vec) const
    {
        double sum = 0.0;
        for (const auto &element : vec)
        {
            sum += element * element;
        }
        return sqrt(sum);
    }

    /**
     * @brief The function normalizes the input vector using L2 norm.
     * @param vec Input vector to normalize
     * @return Normalized vector
     */
    vector<double> normalizeVector(const vector<double> &vec) const
    {
        double norm = calculateL2Norm(vec);
        if (norm < tolerance)
        {
            return vector<double>(vec.size(), 0.0);
        }
        vector<double> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
        {
            result[i] = vec[i] / norm;
        }
        return result;
    }

    /**
     * @brief The function evaluates the dot product of the input vectors.
     * @param vec1 First input vector
     * @param vec2 Second input vector
     * @return Dot product result
     */
    double getDotProduct(const vector<double> &vec1, const vector<double> &vec2) const
    {
        if (vec1.size() != vec2.size())
        {
            throw invalid_argument("Vectors must have the same dimension");
        }
        double result = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i)
        {
            result += vec1[i] * vec2[i];
        }
        return result;
    }

    /**
     * @brief The function evaluates the transpose of the input matrix.
     * @param mat Input matrix
     * @return Transposed matrix
     */
    vector<vector<double>> getTranspose(const vector<vector<double>> &mat) const
    {
        vector<vector<double>> result(mat[0].size(), vector<double>(mat.size()));
        for (size_t i = 0; i < mat.size(); ++i)
        {
            for (size_t j = 0; j < mat[0].size(); ++j)
            {
                result[j][i] = mat[i][j];
            }
        }
        return result;
    }

    /**
     * @brief The function multiplies the input matrices.
     * @param A First input matrix
     * @param B Second input matrix
     * @return Result of matrix multiplication
     */
    vector<vector<double>> multiplyMatrices(const vector<vector<double>> &A,
                                            const vector<vector<double>> &B) const
    {
        if (A[0].size() != B.size())
        {
            throw invalid_argument("Matrix dimensions don't match for multiplication");
        }

        vector<vector<double>> result(A.size(), vector<double>(B[0].size(), 0.0));
        for (size_t i = 0; i < A.size(); ++i)
        {
            for (size_t j = 0; j < B[0].size(); ++j)
            {
                for (size_t k = 0; k < B.size(); ++k)
                {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    /**
     * @brief Generates a Householder matrix to zero out below-diagonal elements of a vector.
     * The Householder matrix H = I - 2vv^T where v is the normalized vector.
     * @param x Input vector to transform
     * @return Householder matrix
     */
    vector<vector<double>> generateHouseholderMatrix(const vector<double> &x) const
    {
        size_t n = x.size();
        double norm = calculateL2Norm(x);

        // If the vector is already zero, return identity matrix
        if (norm < tolerance)
        {
            vector<vector<double>> identity(n, vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i)
            {
                identity[i][i] = 1.0;
            }
            return identity;
        }

        // Create the sign-adjusted norm
        double sign = (x[0] >= 0) ? 1.0 : -1.0;
        double alpha = sign * norm;

        // Create v = x + sign(x₁)‖x‖e₁
        vector<double> v = x;
        v[0] += alpha;

        // Normalize v
        double v_norm = calculateL2Norm(v);
        for (auto &element : v)
        {
            element /= v_norm;
        }

        // Compute H = I - 2vv^T
        vector<vector<double>> H(n, vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                {
                    H[i][j] = 1.0 - 2.0 * v[i] * v[j];
                }
                else
                {
                    H[i][j] = -2.0 * v[i] * v[j];
                }
            }
        }
        return H;
    }

    /**
     * @brief Computes the QR decomposition of the input matrix using Householder transformations.
     * This implementation follows the Householder method which successively transforms
     * the input matrix into an upper triangular matrix while accumulating the
     * orthogonal transformations.
     *
     * @return A pair of matrices {Q, R}, where Q is orthogonal and R is upper triangular.
     */
    pair<vector<vector<double>>, vector<vector<double>>> computeQR()
    {
        size_t n = matrix.size();
        if (n == 0 || matrix[0].size() != n)
        {
            throw invalid_argument("Matrix must be square");
        }

        // Initialize Q as identity matrix
        vector<vector<double>> Q(n, vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i)
        {
            Q[i][i] = 1.0;
        }

        // Initialize R as a copy of input matrix
        vector<vector<double>> R = matrix;

        // Perform the QR decomposition
        for (size_t k = 0; k < n - 1; ++k)
        {
            // Extract the column vector below diagonal
            vector<double> x(n - k);
            for (size_t i = k; i < n; ++i)
            {
                x[i - k] = R[i][k];
            }

            // Generate Householder matrix for this column
            vector<vector<double>> H_small = generateHouseholderMatrix(x);

            // Embed small Householder matrix into nxn matrix
            vector<vector<double>> H(n, vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i)
            {
                H[i][i] = 1.0;
            }
            for (size_t i = k; i < n; ++i)
            {
                for (size_t j = k; j < n; ++j)
                {
                    H[i][j] = H_small[i - k][j - k];
                }
            }

            // Update R and Q matrices
            R = multiplyMatrices(H, R);
            Q = multiplyMatrices(Q, getTranspose(H));
        }

        return {Q, R};
    }
};

int main()
{
    // Example matrix for QR decomposition
    vector<vector<double>> matrix = {
        {1.0, 2.0, 4.0},
        {0.0, 0.0, 5.0},
        {0.0, 3.0, 6.0}};

    QRDecomposition qr(matrix);
    cout << "Original Matrix:" << endl;
    qr.displayMatrix(matrix);
    cout << endl;

    auto [Q, R] = qr.computeQR();

    cout << "Q Matrix (orthogonal):" << endl;
    qr.displayMatrix(Q);
    cout << endl;

    cout << "R Matrix (upper triangular):" << endl;
    qr.displayMatrix(R);
    cout << endl;

    // Verify Q is orthogonal (Q * Q^T should be identity)
    cout << "Verification - Q * Q^T (should be identity):" << endl;
    qr.displayMatrix(qr.multiplyMatrices(Q, qr.getTranspose(Q)));
    cout << endl;

    // Verify decomposition (Q * R should equal original matrix)
    cout << "Verification - Q * R (should equal original matrix):" << endl;
    qr.displayMatrix(qr.multiplyMatrices(Q, R));

    return 0;
}
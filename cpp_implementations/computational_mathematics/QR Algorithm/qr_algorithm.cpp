/**
 * @file qr_algorithm.cpp
 * @brief A script that performs QR decomposition of a matrix using Householder method.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace std;

template <typename T>
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
    const vector<vector<T>> &matrix; // Reference to the original input matrix
    int maxIterations;
    double tolerance;

public:
    // Constructor with member initializer list
    QRDecomposition(const vector<vector<T>> &matrix,
                    int maxIterations = 1000,
                    double tolerance = 1e-10)
        : matrix(matrix), maxIterations(maxIterations), tolerance(tolerance) {}

    // Function to print matrix
    void displayMatrix(const vector<vector<T>> &inputMatrix);

    // Function to calculate the L2 norm
    T calculateL2Norm(const vector<T> &_vector);

    // Function to normalize a vector
    vector<T> normalizeVector(const vector<T> &_vector);

    // Function to get dot product of two vectors
    T getDotProduct(const vector<T> &_vector_A, const vector<T> &_vector_B);

    // Function to get the transpose of a matrix
    vector<vector<T>> getTranspose(const vector<vector<T>> &inputMatrix);

    // Function to multiply two matrices
    vector<vector<T>> multiplyMatrices(const vector<vector<T>> &inputMatrix_A, const vector<vector<T>> &inputMatrix_B);

    // Function to get the householder matrix
    vector<vector<T>> generateHouseholderMatrix(const vector<T> &columnVector);

    // Function to compute Q and R
    pair<vector<vector<T>>, vector<vector<T>>> computeQR(const vector<vector<T>> &matrix);
};

template <typename T>
void QRDecomposition<T>::displayMatrix(const vector<vector<T>> &inputMatrix)
/**
 * @brief The function prints the input matrix.
 */
{
    for (const auto &row : inputMatrix)
    {
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }
}

template <typename T>
T QRDecomposition<T>::calculateL2Norm(const vector<T> &_vector)
/**
 * @brief The function calculates the L2 norm of a vector.
 */
{
    T result = 0;

    for (const auto &element : _vector)
    {
        result += pow(element, 2);
    }
    return sqrt(result);
}

template <typename T>
vector<T> QRDecomposition<T>::normalizeVector(const vector<T> &_vector)
/**
 * @brief The function normalizes the input vector using L2 norm.
 */
{
    T vector_norm = calculateL2Norm(_vector);
    vector<T> normalized_vector(_vector.size());
    for (size_t i = 0; i < _vector.size(); ++i)
    {
        normalized_vector[i] = _vector[i] / vector_norm;
    }
    return normalized_vector;
}

template <typename T>
T QRDecomposition<T>::getDotProduct(const vector<T> &_vector_A, const vector<T> &_vector_B)
/**
 * @brief The function evaluates the dot product of the input vectors.
 */
{
    T result = 0;
    if (_vector_A.size() != _vector_B.size())
    {
        throw invalid_argument("The dimensions must be the same for dot products.");
    }
    for (size_t i = 0; i < _vector_A.size(); ++i)
    {
        result += _vector_A[i] * _vector_B[i];
    }
    return result;
}

template <typename T>
vector<vector<T>> QRDecomposition<T>::getTranspose(const vector<vector<T>> &inputMatrix)
/**
 * @brief The function evaluates the transpose of the input matrix.
 */
{
    size_t num_rows = inputMatrix.size();
    size_t num_cols = inputMatrix[0].size();
    vector<vector<T>> result(num_cols, vector<T>(num_rows, 0));

    for (size_t i = 0; i < num_rows; i++)
    {
        for (size_t j = 0; j < num_cols; j++)
        {
            result[j][i] = inputMatrix[i][j];
        }
    }
    return result;
}

template <typename T>
vector<vector<T>> QRDecomposition<T>::multiplyMatrices(const vector<vector<T>> &inputMatrix_A,
                                                       const vector<vector<T>> &inputMatrix_B)
/**
 * @brief The function multiplies the input matrices.
 */
{
    size_t row_A = inputMatrix_A.size();
    size_t col_A = inputMatrix_A[0].size();
    size_t row_B = inputMatrix_B.size();
    size_t col_B = inputMatrix_B[0].size();

    if (col_A != row_B)
    {
        throw invalid_argument("Matrix multiplication not possible due to dimension mismatch.");
    }

    vector<vector<T>> result(row_A, vector<T>(col_B, 0));

    for (size_t i = 0; i < row_A; ++i)
    {
        for (size_t j = 0; j < col_B; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < col_A; ++k)
            {
                sum += inputMatrix_A[i][k] * inputMatrix_B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

template <typename T>
vector<vector<T>> QRDecomposition<T>::generateHouseholderMatrix(const vector<T> &columnVector)
/**
 * @brief Generates a Householder matrix to zero out below-diagonal elements of a vector.
 */
{
    size_t n = columnVector.size();

    // Compute the L2 norm of the vector
    T norm = calculateL2Norm(columnVector);

    // Create the unit vector e1
    vector<T> e1(n, 0);
    e1[0] = 1;

    // Compute v = x - norm(x) * e1
    vector<T> v(n);
    for (size_t i = 0; i < n; ++i)
    {
        // This condition ensures that only the first element is subtracted from
        // the column vector
        v[i] = columnVector[i] - (i == 0 ? norm : 0);
    }

    // Normalize v
    vector<T> v_normalized = normalizeVector(v);

    // Create the Householder matrix: H = I - 2 * v * v^T
    vector<vector<T>> H(n, vector<T>(n, 0)); // Initialize H as an identity matrix

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            // Subtracting the elements from the diagonal matrix.
            if (i == j)
                H[i][j] = 1 - 2 * v_normalized[i] * v_normalized[j];
            else
                H[i][j] = -2 * v_normalized[i] * v_normalized[j];
        }
    }

    return H;
}
template <typename T>
pair<vector<vector<T>>, vector<vector<T>>>
QRDecomposition<T>::computeQR(const vector<vector<T>> &matrix)
/**
 * @brief Computes the QR decomposition of the input matrix using Householder transformations.
 *
 * @return A pair of matrices {Q, R}, where Q is orthogonal and R is upper triangular.
 */
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // Copy of the input matrix to modify
    vector<vector<T>> A = matrix;

    // Initialising Q as an identity matrix
    vector<vector<T>> Q = vector<vector<T>>(rows, vector<T>(rows, 0));

    // Initialize Q as an identity matrix
    for (size_t i = 0; i < rows; ++i)
    {
        Q[i][i] = 1;
    }

    for (size_t k = 0; k < cols && k < rows - 1; ++k)
    {
        // Extract the k-th column below the diagonal
        vector<T> columnVector(rows - k);
        for (size_t i = k; i < rows; ++i)
        {
            columnVector[i - k] = A[i][k];
        }

        // Generate the Householder matrix for the k-th column
        vector<vector<T>> H_k = vector<vector<T>>(rows, vector<T>(rows, 0));

        vector<vector<T>> H_local = generateHouseholderMatrix(columnVector);

        // Embed the H_local into a larger identity matrix (H_k)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < rows; ++j)
            {
                if (i < k || j < k)

                    // Preserving the  identity matrix
                    H_k[i][j] = (i == j) ? 1 : 0;
                else
                    H_k[i][j] = H_local[i - k][j - k]; // Embed H_local
            }
        }

        // Update A: A = H_k * A
        A = multiplyMatrices(H_k, A);

        // Update Q: Q = Q * H_k^T (accumulate transformations)
        Q = multiplyMatrices(Q, getTranspose(H_k));
    }

    return {Q, A}; // Q is orthogonal, A is now upper triangular (R)
}

int main()
{
    vector<vector<int>> matrix = {{1, 2, 1},
                                  {2, 4, 3},
                                  {4, 5, 2}};

    QRDecomposition<int> instance(matrix);

    cout << "Original Matrix:" << endl;
    instance.displayMatrix(matrix);

    auto [Q, R] = instance.computeQR(matrix);

    cout << "Matrix Q:" << endl;
    instance.displayMatrix(Q);

    cout << "Matrix R:" << endl;
    instance.displayMatrix(R);

    return 0;
}

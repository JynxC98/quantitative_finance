/**
 * @file: Singular_Value_Decomposition.cpp
 * @brief: A script to perform Singular Value Decomposition of a matrix.
 * @author: Harsh Parikh
 */

#include <iostream>
#include <vector>

using namespace std;

template <typename T>

class SingularValueDecomposition
/**
 * @brief This class implements the Singular Value Decomposition (SVD) of a matrix.
 * SVD is a matrix factorization technique that decomposes a given matrix A into three matrices:
 *
 * A = U * Σ * V^T
 *
 * where:
 * - A: The original matrix (m x n).
 * - U: An orthogonal matrix of left singular vectors (m x m).
 * - Σ: A diagonal matrix of singular values (m x n) where the singular values are non-negative and arranged in decreasing order along the diagonal.
 * - V^T: The transpose of an orthogonal matrix of right singular vectors (n x n).
 *
 * Singular Value Decomposition is a powerful tool in linear algebra used for a variety of applications such as:
 * - Solving linear systems and least-squares problems.
 * - Principal Component Analysis (PCA) for dimensionality reduction.
 * - Matrix approximation, noise reduction, and image compression.
 * The resulting decomposition is:
 * A = U * Σ * V^T
 *
 * where U, Σ, and V^T are matrices such that:
 * - U contains the left singular vectors.
 * - Σ contains the singular values.
 * - V^T contains the right singular vectors.
 *
 * **Computational Complexity**:
 * - Time complexity: O(min(m^2n, mn^2)), where m is the number of rows and n is the number of columns.
 * - Space complexity: O(mn) for storing matrices U, Σ, and V.
 *
 *
 * **Notes**:
 * - The matrix Σ contains singular values, which are the square roots of the eigenvalues of the matrix A^T * A.
 * - SVD is often used in problems where the matrix may be ill-conditioned or rank-deficient.
 *
 * **Algorithm Overview**:
 * 1. Compute the eigenvalues and eigenvectors of the matrix A^T * A.
 * 2. Construct the matrix V from the eigenvectors.
 * 3. Compute A * V to find U, and form the diagonal matrix Σ from the singular values.
 *
 * @return A tuple containing the matrices {U, Σ, V^T}, where:
 *   - U is the matrix of left singular vectors.
 *   - Σ is the diagonal matrix of singular values.
 *   - V^T is the transpose of the matrix of right singular vectors.
 */

{

private:
    const vector<vector<T>> &matrix;

public:
    // Initialising the class with the input matrix.
    SingularValueDecomposition(const vector<vector<T>> &matrix) : matrix(matrix) {}

    // Function used to print the input matrix
    void displayMatrix(const vector<vector<T>> &inputMatrix);

    // Function used to get the transpose of the input matrix.
    vector<vector<T>> getTranspose(const vector<vector<T>> &inputMatrix);

    // Function used to multiply two matrices.
    vector<vector<T>> multiplyMatrices(const vector<vector<T>> &inputMatrix_A, const vector<vector<T>> &inputMatrix_B);

    // Function used to calculate eigen values and eigen vectors of the input matrix
    pair<vector<T>, vector<vector<T>>> getEigenValuesandEigenVectors(const vector<vector<T>> &inputMatrix);
};

template <typename T>
void SingularValueDecomposition<T>::displayMatrix(const vector<vector<T>> &inputMatrix)
{
    /**
     * @brief The function prints the input matrix.
     */
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
vector<vector<T>> getTranspose(const vector<vector<T>> &inputMatrix)
{
}

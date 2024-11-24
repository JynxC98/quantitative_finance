/**
 * @file cholskey_decomposition.cpp
 * @brief A script that performs Cholskey decomposition of a matrix.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

template <typename T>

class CholskeyDecomposition
/**
 * @brief Cholesky decomposition is a matrix factorisation technique
 * that converts a positive semidefinite square matrix into the form:
 *
 * A = L * L^T,
 * where,
 * A: The original positive semidefinite square matrix
 * L: Lower triangular matrix with positive diagonal entries
 * L^T: Transpose of the lower triangular matrix
 *
 * This method computes the Cholesky decomposition of the input matrix.
 * If the decomposition fails (e.g., if the matrix is not positive semidefinite),
 * the function indicates an error.
 *
 * @param matrix: Reference to the original square matrix to decompose.
 * @return A lower triangular matrix L and its transpose L^T if decomposition succeeds.
 */

{

    // Initialising the input matrix.
private:
    const vector<vector<T>> &matrix;

public:
    // Constructor to assign the value of matrix.
    CholskeyDecomposition(const vector<vector<T>> &matrix) : matrix(matrix) {}

    // Function to print a matrix.
    void displayMatrix(const vector<vector<T>> &inputMatrix);

    // Function to evaluate the symmetricity of a given matrix.
    bool isSymmetric();

    // Function to calculate the transpose of a given matrix.
    vector<vector<T>> getTranspose(const vector<vector<T>> &inputMatrix);

    // Function to calculate the lower triangular matrix and its transpose.
    pair<vector<vector<T>>, vector<vector<T>>> getRequiredMatrices();
};

template <typename T>
void CholskeyDecomposition<T>::displayMatrix(const vector<vector<T>> &inputMatrix)
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
bool CholskeyDecomposition<T>::isSymmetric()
/**
 * @brief The function checks the symmetricity of the matrix.
 */

{
    int num_rows = matrix.size();
    int num_columns = matrix[0].size();

    if (num_rows != num_columns)
    {
        throw invalid_argument("The matrix is not a square matrix hence asymmetric");
    }
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (matrix[i][j] != matrix[j][i])
            {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
vector<vector<T>> CholskeyDecomposition<T>::getTranspose(const vector<vector<T>> &inputMatrix)
/**
 * @brief The function evaluates the transpose of the input matrix.
 */
{
    int num_rows = inputMatrix.size();
    int num_columns = inputMatrix[0].size();

    vector<vector<T>> result(num_columns, vector<T>(num_rows, 0)); // Create a transposed matrix with swapped dimensions

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_columns; j++)
        {
            result[j][i] = inputMatrix[i][j];
        }
    }
    return result;
}

template <typename T>
pair<vector<vector<T>>, vector<vector<T>>>
CholskeyDecomposition<T>::getRequiredMatrices()
/**
 * @brief The function calculates the lower triangular matrix (L) and its transpose (L^T)
 * obtained from Cholesky Decomposition.
 *
 * Cholesky Decomposition is a matrix factorization technique that expresses a
 * symmetric positive-definite matrix A as:
 *
 * A = L * L^T
 *
 * where:
 * - A: The original symmetric positive-definite matrix.
 * - L: A lower triangular matrix with strictly positive diagonal entries.
 * - L^T: The transpose of the lower triangular matrix L.
 *
 * Symbolic Example:
 * Let the original matrix A be:
 * A = [a11  a12  a13]
 *     [a12  a22  a23]
 *     [a13  a23  a33]
 *
 * The resulting matrix L is:
 * L = [l11    0      0   ]
 *     [l21    l22    0   ]
 *     [l31    l32    l33 ]
 *
 * The decomposition guarantees that:
 * 1. A = L * L^T
 * 2. The diagonal elements of L are positive.
 * 3. The decomposition is only valid for symmetric positive-definite matrices.
 *
 * Algorithm Overview:
 * For each element L[i][j] in the lower triangular matrix:
 * - If i == j (diagonal elements), compute:
 *       L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k = 0 to i-1))
 * - If i > j (off-diagonal elements), compute:
 *       L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k] for k = 0 to j-1)) / L[j][j]
 * - If i < j (upper triangle), set L[i][j] = 0
 *
 * Computational Complexity:
 * - Time complexity: O(n^3), where n is the size of the matrix.
 * - Space complexity: O(n^2), due to storage of matrix L.
 *
 * Preconditions:
 * 1. The input matrix A must be symmetric (A[i][j] == A[j][i]).
 * 2. The matrix must be positive-definite (all eigenvalues > 0).

 * @return A pair of matrices {L, L^T}, where L is the lower triangular matrix
 *         and L^T is its transpose.
 */

{
    if (!isSymmetric())
    {
        throw invalid_argument("Cholskey decomposition is not possible as the matrix is asymettric");
    }
    int num_rows = matrix.size();

    vector<vector<T>> L(num_rows, vector<T>(num_rows, 0));

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_rows; j++)
        {

            T sum = 0;

            // Condition for the diagonal elements.
            if (i == j)
            {

                for (int k = 0; k < j; k++)
                {
                    sum += pow(L[j][k], 2);
                }
                L[j][j] = sqrt(matrix[j][j] - sum);
            }

            // Condition for non-diagonal elements.
            else
            {
                // Condition to ensure computation of lower triangular elements.
                if (i > j)
                {

                    for (int k = 0; k < j; k++)
                    {
                        sum += L[j][k] * L[i][k];
                    }
                    L[i][j] = (matrix[i][j] - sum) / (L[j][j]);
                }

                // Since we are working on a lower triangular matrix, the elements
                // above the diagonal would be zero.
                else
                {
                    L[i][j] = 0;
                }
            }
        }
    }
    vector<vector<T>> L_transpose = getTranspose(L);
    return make_pair(L, L_transpose);
}

int main()
{
    vector<vector<int>> matrix = {{4, 12, -16},
                                  {12, 37, -43},
                                  {-16, -43, 98}};

    CholskeyDecomposition<int> instance(matrix);
    auto [L, L_transpose] = instance.getRequiredMatrices();
    cout << "Printing the lower triangular matrix" << endl;
    instance.displayMatrix(L);
    cout << "Printing the transpose of the lower triangular matrix" << endl;
    instance.displayMatrix(L_transpose);
    return 0;
}
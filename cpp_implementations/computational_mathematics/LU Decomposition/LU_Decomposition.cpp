/**
 * @file lu_decomposition.cpp
 * @brief A script that performs LU Decomposition of a matrix.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>

using namespace std;

template <typename T>
class LUDecomposition
/**
 * @brief LU decomposition is a matrix factorisation technique that
 * converts a square matrix in the following form:
 *
 * matrix = L * U,
 * @where,
 * matrix: The original square matrix
 * L: Lower triangular matrix
 * U: Upper triangular matrix
 * * **Symbolic Example**:
 * Let the original matrix A be:
 * A = [a11  a12  a13]
 *     [a21  a22  a23]
 *     [a31  a32  a33]
 *
 * The resulting matrices are:
 * L = [1      0      0   ]
 *     [l21    1      0   ]
 *     [l31    l32    1   ]
 *
 * U = [u11    u12    u13 ]
 *     [0      u22    u23 ]
 *     [0      0      u33 ]
 *
 * **Algorithm Overview**:
 * 1. For each element in the matrix:
 *   - Compute elements of U:
 *       U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k = 0 to i-1)
 *       (for all i <= j)
 *   - Compute elements of L:
 *       L[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k = 0 to j-1)) / U[j][j]
 *       (for all i > j)
 *   - Set L[i][j] = 1 for all diagonal elements (i == j).
 */
{
private:
    const vector<vector<T>> &matrix;

public:
    // Constructor to assign the value of matrix.
    LUDecomposition(const vector<vector<T>> &matrix) : matrix(matrix) {}

    void displayMatrix();
    pair<vector<vector<T>>, vector<vector<T>>> getRequiredMatrices();
};

template <typename T>
void LUDecomposition<T>::displayMatrix()
{
    for (const auto &row : matrix)
    {
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }
}

template <typename T>
pair<vector<vector<T>>,
     vector<vector<T>>>

LUDecomposition<T>::getRequiredMatrices()
/**
 * @brief This function calculates the lower triangular matrix (L) and upper triangular
 * matrix (U) obtained from LU decomposition using the Doolittle Algorithm.
 *
 * @cite GeeksforGeeks (https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/)
 * @cite Wikipedia (https://en.wikipedia.org/wiki/LUdecomposition)
 *
 * @return A pair of matrices {L, U}, where L is a lower triangular matrix with 1s on the diagonal,
 *         and U is an upper triangular matrix.
 */

{
    int num_rows = matrix.size();
    int num_columns = matrix[0].size();

    if (num_rows != num_columns)
    {
        throw invalid_argument("LU decomposition not possible due to dimension mismatch");
    }
    if (matrix.empty() || matrix[0].empty())
    {
        throw invalid_argument("Matrix cannot be empty.");
    }

    // Initialising the upper triangular matrix
    vector<vector<T>> L(num_rows, vector<T>(num_rows, 0));

    // Initialising the lower triangular matrix
    vector<vector<T>> U(num_rows, vector<T>(num_rows, 0));

    for (int i = 0; i < num_rows; i++)
    {
        // Working on upper triangular matrix
        for (int j = 0; j < num_rows; j++)
        {

            if (i == 0)
            {
                U[i][j] = matrix[i][j];
                continue;
            }

            T sum = 0;
            for (int k = 0; k < i; k++)
            {
                sum += L[i][k] * U[k][j];
            }
            U[i][j] = matrix[i][j] - sum;
        }

        // Working on lower triangular matrix
        for (int j = i; j < num_rows; j++)
        {
            if (i == j)
                L[i][i] = 1; // Diagonal elements of L are 1
            else
            {
                T sum = 0;
                for (int k = 0; k < i; k++)
                {
                    sum += L[j][k] * U[k][i];
                }

                L[j][i] = (matrix[j][i] - sum) / U[i][i];
            }
        }
    }
    return {L, U};
}

int main()
{
    vector<vector<int>> matrix = {{2, -1, -2},
                                  {-4, 6, 3},
                                  {-4, -2, 8}};

    LUDecomposition<int> instance(matrix);

    instance.displayMatrix();

    auto [L, U] = instance.getRequiredMatrices();

    cout << "Lower triangular matrix (L):" << endl;
    for (const auto &row : L)
    {
        for (const auto &element : row)
            cout << element << " ";
        cout << endl;
    }

    cout << "Upper triangular matrix (U):" << endl;
    for (const auto &row : U)
    {
        for (const auto &element : row)
            cout << element << " ";
        cout << endl;
    }

    return 0;
}

/**
 * @file LU_Decomposition.cpp
 * @brief A script that performs LU Decomposition of two matrices.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>

using namespace std;

template <typename T>
class LU_Decomposition
/**
 * @brief LU decomposition is a matrix factorisation technique that
 * converts a square matrix in the following form:
 *
 * A = L * U,
 * @where,
 * A: The original square matrix
 * L: Lower triangular matrix
 * U: Upper triangular matrix
 *
 * Initialises the class with the original square matrix.
 *
 * @param A Reference to the first matrix.
 * @param B Reference to the second matrix.
 */
{
private:
    const vector<vector<T>> &matrix;

public:
    // Constructor to assign the value of matrix.
    LU_Decomposition(const vector<vector<T>> &matrix) : matrix(matrix) {}

    void displayMatrix();
    pair<vector<vector<T>>, vector<vector<T>>> getRequiredMatrices();
};

template <typename T>
void LU_Decomposition<T>::displayMatrix()
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

LU_Decomposition<T>::getRequiredMatrices()
/**
 * @brief The function returns the lower triangular (L) and upper triangular (U) matrices
 * obtained from LU decomposition.
 *
 * @cite https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
 *
 * LU decomposition is a matrix factorization technique that expresses a square matrix A
 * as the product of a lower triangular matrix L and an upper triangular matrix U
 *
 * A = L * U
 *
 * where,
 * A: The original square matrix
 * L: Lower triangular matrix
 * U: Upper triangular matrix
 *
 * Symbolic Example:
 * Let the original matrix A be:
 * A = [a11  a12  a13]
 *     [a21  a22  a23]
 *     [a31  a32  a33]
 *
 * The resulting matrices are:
 * L = [111      0      0   ]
 *     [l21    122      0   ]
 *     [l31    l32    133   ]
 *
 * U = [u11    u12    u13 ]
 *     [0      u22    u23 ]
 *     [0      0      u33 ]
 *
 * @return A pair of matrices {L, U}, where L is lower triangular and U is upper triangular.
 */

{
    int num_rows = matrix.size();
    int num_columns = matrix[0].size();

    if (num_rows != num_columns)
    {
        throw invalid_argument("LU decomposition not possible due to dimension mismatch");
    }

    // Initialising the upper triangular matrix
    vector<vector<T>> L(num_rows, vector<T>(num_rows, 0));

    // Initialising the lower triangular matrix
    vector<vector<T>> U(num_rows, vector<T>(num_rows, 0));

    for (int i = 0; i < num_rows; i++)
    {
        // Working on lower triangular matrix
        for (int j = 0; j < num_rows; j++)
        {

            T sum = 0;
            for (int k = 0; k < i; k++)
            {
                sum += L[i][k] * U[k][j];
            }
            U[i][j] = matrix[i][j] - sum;
        }
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

    LU_Decomposition<int> instance(matrix);

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

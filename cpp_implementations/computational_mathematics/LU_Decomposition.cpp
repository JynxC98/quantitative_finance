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
 * @brief The function returns left triangular and right triangular matrix.
 *
 * A = L * U,
 * @where,
 * A: The original square matrix
 * L: Lower triangular matrix
 * U: Upper triangular matrix
 *
 */
{
    int num_rows = matrix.size();
    int num_columns = matrix[0].size();

    if (num_rows != num_columns)
    {
        throw invalid_argument("LU decomposition not possible due to dimension mismatch");
    }

    // Initialising the upper triangular matrix
    vector<vector<T>> upper_triangular;

    // Initialising the lower triangular matrix
    vector<vector<T>> lower_triangular;
}

int main()
{
    vector<vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    LU_Decomposition<int> instance(matrix);

    instance.displayMatrix();

    return 0;
}

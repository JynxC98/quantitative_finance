/**
 * @file MatMul.cpp
 * @brief A script to multiply two matrices using a templated class.
 */

#include <iostream>
#include <vector>

using namespace std;

/**
 * @class MatMul
 * @brief A templated class for matrix multiplication.
 *
 * This class provides functionalities for multiplying two matrices of
 * compatible dimensions and displaying the matrices and their product.
 *
 * @tparam T The data type of the matrix elements (e.g., int, float, double).
 */
template <typename T>
class MatMul
{
private:
    const vector<vector<T>> &matrix_A; ///< Reference to the first input matrix.
    const vector<vector<T>> &matrix_B; ///< Reference to the second input matrix.

public:
    /**
     * @brief Constructor for the MatMul class.
     *
     * Initializes the class with two matrices to be multiplied.
     *
     * @param A Reference to the first matrix.
     * @param B Reference to the second matrix.
     */
    MatMul(const vector<vector<T>> &A, const vector<vector<T>> &B);

    /**
     * @brief Displays the two matrices (Matrix A and Matrix B).
     *
     * Prints the elements of the input matrices to the console.
     */
    void displayMatrices();

    /**
     * @brief Multiplies the two matrices.
     *
     * Performs matrix multiplication, ensuring the dimensions of the matrices
     * are compatible.
     *
     * @return A 2D vector representing the resulting matrix product.
     * @throws std::invalid_argument if the matrices' dimensions are not compatible.
     */
    vector<vector<T>> multiply();

    /**
     * @brief Displays the resulting matrix product.
     *
     * Computes the product of the two matrices and prints the result to the console.
     */
    void displayProduct();
};

template <typename T>
MatMul<T>::MatMul(const vector<vector<T>> &A, const vector<vector<T>> &B) : matrix_A(A), matrix_B(B) {}

template <typename T>
void MatMul<T>::displayMatrices()
{
    cout << "Displaying Matrix A:" << endl;
    for (const auto &row : matrix_A)
    {
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }

    cout << "Displaying Matrix B:" << endl;
    for (const auto &row : matrix_B)
    {
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }
}

template <typename T>
vector<vector<T>> MatMul<T>::multiply()
{
    int row_A = matrix_A.size();
    int col_A = matrix_A[0].size();
    int row_B = matrix_B.size();
    int col_B = matrix_B[0].size();

    if (col_A != row_B)
    {
        throw invalid_argument("Matrix multiplication not possible due to dimension mismatch");
    }

    vector<vector<T>> result(row_A, vector<T>(col_B, 0));

    for (int i = 0; i < row_A; i++)
    {
        for (int j = 0; j < col_B; j++)
        {
            T sum = 0;
            for (int k = 0; k < col_A; k++)
            {
                sum += matrix_A[i][k] * matrix_B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

template <typename T>
void MatMul<T>::displayProduct()
{
    vector<vector<T>> result = multiply();

    cout << "The product of two matrices is:" << endl;
    for (const auto &row : result)
    {
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }
}

int main()
{
    vector<vector<int>> matrix_A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    vector<vector<int>> matrix_B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatMul<int> matmul(matrix_A, matrix_B);

    matmul.displayMatrices();
    matmul.displayProduct();

    return 0;
}

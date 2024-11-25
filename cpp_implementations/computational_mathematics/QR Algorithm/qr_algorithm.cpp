/**
 * @file qr_algorithm.cpp
 * @brief A script that performs QR decomposition of a matrix.
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>

using namespace std;

template <typename T>
class QRDecomposition
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
    QRDecomposition(const vector<vector<T>> &matrix,
                    int maxIterations = 1000,
                    double tolerance = 1e-10)
        : matrix(matrix), maxIterations(maxIterations), tolerance(tolerance) {}

    void displayMatrix(const vector<vector<T>> &inputMatrix);
    T calculate_L2_norm(const vector<vector<T>> _vector);
    vector<vector<T>> getTranspose(const vector<vector<T>> &inputMatrix);
    vector<vector<T>>
    mutliplyMatrices(const vector<vector<T>> inputMatrix_A, const vector<vector<T>> inputMatrix_B);
    T getDotProduct(const vector<T> _vector_A, const vector<T> _vectorB);
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
T QRDecomposition<T>::calculate_L2_norm(const vector<vector<T>> _vector)
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
vector<vector<T>> QRDecomposition<T>::getTranspose(const vector<vector<T>> &inputMatrix)
/**
 * @brief The function evaluates the transpose of the input matrix.
 */
{
    int num_rows = inputMatrix.size();
    int num_cols = inputMatrix[0].size();
    vector<vector<T>> result(num_cols, vector<T>(num_rows, 0));

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            result[j][i] = inputMatrix[i][j];
        }
    }
    return result;
}

template <typename T>
vector<vector<T>> QRDecomposition<T>::
    mutliplyMatrices(const vector<vector<T>> inputMatrix_A, const vector<vector<T>> inputMatrix_B)
/**
 * @brief The function multiplies the input matrices.
 */
{
    int row_A = inputMatrix_A.size();
    int col_A = inputMatrix_A[0].size();
    int row_B = inputMatrix_B.size();
    int col_B = inputMatrix_B[0].size();

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
T QRDecomposition<T>::getDotProduct(vector<T> _vector_A, vector<T> _vector_B)
{
    T result = 0;
    if (_vector_A.size() != _vector_B.size())
    {
        throw invalid_argument("The dimensions must be the same for dot products.")
    }
    for (int i = 0; i < _vector_A.size(); i++)
    {
        result += _vector_A[i] * _vector_B[i];
    }
    return result;
}
int main()
{
    vector<vector<int>> matrix = {{1, 2, 1},
                                  {2, 4, 3},
                                  {4, 5, 2}};

    QRDecomposition<int> instance(matrix);
    cout << "Displaying the matrix" << endl;
    instance.displayMatrix(matrix);
    cout << "Displaying the transpose" << endl;
    instance.displayMatrix(instance.getTranspose(matrix));
    return 0;
}

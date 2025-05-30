/**
 * @file qr_decomposition.cpp
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
/**
 * @note Template class is not required as QR decomposition relies extensively
 * on floating-point precision.
 *
 * @brief QR algorithm is an iterative method used to compute the eigenvalues
 * and eigenvectors of a square matrix by factorizing the matrix into:
 *
 * A = Q * R,
 * where,
 * A: The original square matrix
 * Q: Orthogonal matrix
 * R: Upper triangular matrix
 *
 * @param matrix: Reference to the square matrix to process.
 * @param maxIterations: Maximum number of iterations for the algorithm.
 * @param tolerance: Convergence threshold for the off-diagonal entries.
 */
{
private:
    vector<vector<double>> &matrix;
    int maxIterations;
    double tolerance;

public:
    QRDecomposition(vector<vector<double>> &matrix,
                    int maxIterations = 100000, double tolerance = 1e-4)
        : matrix(matrix), maxIterations(maxIterations), tolerance(tolerance) {}

    // Function to display the matrix.
    void displayMatrix(const vector<vector<double>> &matrix);

    // Function to calculate L2 norm.
    double getL2Norm(const vector<double> &_vector);

    // Function to normalise a vector
    vector<double> normaliseVector(const vector<double> &_vector);

    // Function to get the transpose of the input matrix.
    vector<vector<double>> getTranspose(const vector<vector<double>> &matrix);

    // Function to calculate matrix multiplication
    vector<vector<double>> multiplyMatrices(const vector<vector<double>> &matrix1,
                                            const vector<vector<double>> &matrix2);

    // Function to generate the Householder matrix
    vector<vector<double>> generateHouseholderMatrix(const vector<double> &columnVector);

    // Function to get Q and R matrices
    pair<vector<vector<double>>, vector<vector<double>>> computeQR(const vector<vector<double>> &matrix);

    // Function to check whether the matrix is approximately upper triangular
    bool isUpperTriangular(const vector<vector<double>> &matrix);

    // Function to get Eigen Values and Eigen Vectors
    pair<vector<double>, vector<vector<double>>> getEigenValuesandEigenVectors(const vector<vector<double>> &matrix);

    // Function to display the values of eigen values and their corresponding eigen vectors.
    void displayEigenValuesandEigenVectors(const vector<double> &eigenValues,
                                           const vector<vector<double>> &eigenVectors);
};

/**
 * @brief The function prints the input matrix with proper formatting.
 * @param matrix The matrix to be displayed
 */

void QRDecomposition::displayMatrix(const vector<vector<double>> &matrix)
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
 * @param _vector Input vector
 * @return L2 norm of the vector
 */

double QRDecomposition::getL2Norm(const vector<double> &_vector)
{
    double sum = 0;
    for (const auto &element : _vector)
    {
        sum += pow(element, 2);
    }
    return sqrt(sum);
}

/**
 * @brief The function normalises the input vector using L2 norm.
 * @param _vector Input vector to normalise
 * @return Normalised vector
 */

vector<double> QRDecomposition ::normaliseVector(const vector<double> &_vector)
{
    double norm = getL2Norm(_vector);
    if (norm < tolerance)
    {
        return vector<double>(_vector.size(), 0.0);
    }
    vector<double> result(_vector.size());
    for (size_t i = 0; i < _vector.size(); ++i)
    {
        result[i] = _vector[i] / norm;
    }
    return result;
}
/**
 * @brief The function evaluates the transpose of the input matrix.
 * @param matrix Input matrix
 * @return Transposed matrix
 */
vector<vector<double>> QRDecomposition ::getTranspose(const vector<vector<double>> &matrix)
{
    vector<vector<double>> result(matrix[0].size(), vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[0].size(); ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
/**
 * @brief The function multiplies the input matrices.
 * @param matrix1 First input matrix
 * @param matrix2 Second input matrix
 * @return Result of matrix multiplication
 */
vector<vector<double>> QRDecomposition::multiplyMatrices(const vector<vector<double>> &matrix1,
                                                         const vector<vector<double>> &matrix2)
{
    // Check if the number of columns of matrix1 matches the number of rows of matrix2
    if (matrix1[0].size() != matrix2.size())
    {
        throw invalid_argument("Matrix dimensions don't match for multiplication");
    }

    // Initialise the result matrix with appropriate dimensions
    vector<vector<double>> result(matrix1.size(), vector<double>(matrix2[0].size(), 0.0));

    // Perform matrix multiplication
    for (size_t i = 0; i < matrix1.size(); ++i)
    {
        for (size_t j = 0; j < matrix2[0].size(); ++j)
        {
            for (size_t k = 0; k < matrix2.size(); ++k)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}
/**
 * @brief Generates a Householder matrix to zero out below-diagonal elements of a vector.
 * The Householder matrix H = I - 2vv^T where v is the normalised column vector.
 * @param x Input vector to transform
 * @return Householder matrix
 */
vector<vector<double>> QRDecomposition::generateHouseholderMatrix(const vector<double> &columnVector)
{
    size_t num_elements = columnVector.size();
    double norm = getL2Norm(columnVector);

    // If norm is smaller than tolerance, return the identity matrix
    if (norm < tolerance)
    {
        vector<vector<double>> identity(num_elements, vector<double>(num_elements, 0.0));
        for (size_t i = 0; i < num_elements; ++i)
        {
            identity[i][i] = 1.0;
        }
        return identity;
    }

    // Creating the sign adjusted norm
    double sign = (columnVector[0] >= 0) ? 1.0 : -1.0;
    double alpha = sign * norm;

    // Creating v = x + sign(x₁)‖x‖e₁
    vector<double> v = columnVector;
    v[0] += alpha;

    // Normalising v
    double v_norm = getL2Norm(v);
    for (auto &element : v)
    {
        element /= v_norm;
    }

    // Computing the Householder's matrix `H`.
    vector<vector<double>> H(num_elements, vector<double>(num_elements, 0));

    for (size_t i = 0; i < num_elements; i++)
    {
        for (size_t j = 0; j < num_elements; j++)
        {
            // Case 1: Diagonal elements
            if (i == j)
            {
                H[i][j] = 1.0 - 2.0 * v[i] * v[j];
            }
            // Case 2: Non-diagonal elements
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
pair<vector<vector<double>>, vector<vector<double>>> QRDecomposition::
    computeQR(const vector<vector<double>> &matrix)
{
    size_t num_elements = matrix.size();
    if (num_elements == 0 || matrix[0].size() != num_elements)
    {
        throw invalid_argument("Matrix must be square");
    }

    // Initialising Q as an identity matrix.
    vector<vector<double>> Q(num_elements, vector<double>(num_elements, 0.0));
    for (size_t i = 0; i < num_elements; i++)
    {
        Q[i][i] = 1;
    }

    // Initialising R as the copy of the matrix
    vector<vector<double>> R = matrix;

    // Performing the QR decomposition.

    for (size_t k = 0; k < num_elements - 1; k++)
    {
        // Extract the column vector below the diagonal
        vector<double> x(num_elements - k);
        for (size_t i = k; i < num_elements; i++)
        {
            x[i - k] = R[i][k];
        }

        // Generate Householder matrix for this subvector
        vector<vector<double>> H_small = generateHouseholderMatrix(x);

        // Create the full-size Householder matrix
        vector<vector<double>> H(num_elements, vector<double>(num_elements));

        // Initialise `H` as an identity matrix
        for (size_t i = 0; i < num_elements; i++)
        {
            for (size_t j = 0; j < num_elements; j++)
            {
                if (i == j)
                    H[i][j] = 1.0;
                else
                    H[i][j] = 0.0;
            }
        }

        // Embed the smaller Householder matrix
        for (size_t i = k; i < num_elements; i++)
        {
            for (size_t j = k; j < num_elements; j++)
            {
                H[i][j] = H_small[i - k][j - k];
            }
        }

        // Update R and Q with correct signs
        R = multiplyMatrices(H, R);               // Update R as H * R
        Q = multiplyMatrices(Q, getTranspose(H)); // Update Q as Q * H^T
    }

    return {Q, R};
}
/**
 * @brief Evalues whether the input matrix is in the upper triangular based on the
 * predetermined tolerance limit.
 * @param: matrix
 * @return A boolean number where
 * 1: The matrix is approximately upper triangular
 * 0: The matrix is not upper triangular
 */
bool QRDecomposition::isUpperTriangular(const vector<vector<double>> &matrix)
{
    // Since the input matrix will always be a square matrix, there
    // is no need to explicitely check whether the number of rows
    // will be equal to the number of columns.

    size_t num_elements = matrix.size();
    for (size_t i = 0; i < num_elements; ++i)
    {
        for (size_t j = 0; j < i; ++j) // Only check below the diagonal
        {
            if (abs(matrix[i][j]) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief The function calculates eigen values and eigen vectors using QR Decomposition.
 * The function implements the following method to calculate the eigen values
 * A_k = Q_k * R_k
 * A_k+1 = R_k * Q_k
 * @return A boolean number where
 * 1: The matrix is approximately diagonal
 * 0: The matrix is not diagonal
 */
pair<vector<double>, vector<vector<double>>> QRDecomposition::
    getEigenValuesandEigenVectors(const vector<vector<double>> &matrix)
{
    size_t num_elements = matrix.size();
    vector<vector<double>> matrix_new = matrix;

    // Initialising eigenVectors as identity matrix
    vector<vector<double>> eigenVectors(num_elements, vector<double>(num_elements, 0));
    for (int i = 0; i < num_elements; ++i)
    {
        eigenVectors[i][i] = 1.0;
    }

    // Iterative QR algorithm
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Perform QR decomposition
        auto [Q, R] = computeQR(matrix_new);

        // Accumulate eigenvectors - right multiplication
        vector<vector<double>> newEigenVectors = multiplyMatrices(eigenVectors, Q);
        eigenVectors = newEigenVectors;

        // Update matrix_new to be R * Q for the next iteration
        matrix_new = multiplyMatrices(R, Q);

        // Check convergence - more robust way
        if (isUpperTriangular(matrix_new))
        {
            break;
        }
    }

    // Normalise the eigenvectors
    for (int j = 0; j < num_elements; ++j)
    {
        vector<double> eigenvector(num_elements);
        for (int i = 0; i < num_elements; ++i)
        {
            eigenvector[i] = eigenVectors[i][j];
        }

        eigenvector = normaliseVector(eigenvector);

        for (int i = 0; i < num_elements; ++i)
        {
            eigenVectors[i][j] = eigenvector[i];
        }
    }

    // Extracting eigenvalues from the diagonal of matrix_new
    vector<double> eigen_values;
    for (size_t i = 0; i < num_elements; ++i)
    {
        eigen_values.push_back(matrix_new[i][i]);
    }

    return {eigen_values, eigenVectors};
}
/**
 * @brief The function displays the eigen values and its corresponding eigen vectors.

 * @param: eigenValues and eigenVectors.
 */
void QRDecomposition::displayEigenValuesandEigenVectors(const vector<double> &eigenValues,
                                                        const vector<vector<double>> &eigenVectors)
{
    for (size_t i = 0; i < eigenValues.size(); ++i)
    {
        cout << "Eigenvalue " << fixed << setprecision(6) << setw(12) << eigenValues[i] << " : ";
        cout << "Eigenvector: [ ";
        for (size_t j = 0; j < eigenVectors.size(); ++j)
        {
            // eigenVectors[j][i] is the i-th eigenvector
            cout << fixed << setprecision(6) << setw(12) << eigenVectors[j][i] << " ";
        }
        cout << "]" << endl;
    }
}

int main()
{
    // Example matrix for QR decomposition
    vector<vector<double>> matrix = {
        {-1.0, 3.0, 5.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}};

    QRDecomposition qr(matrix);
    cout << "Original Matrix:" << endl;
    qr.displayMatrix(matrix);
    cout << endl;

    auto [Q, R] = qr.computeQR(matrix);

    cout << "Q Matrix (orthogonal):" << endl;
    qr.displayMatrix(Q);
    cout << endl;

    cout << "R Matrix (upper triangular):" << endl;
    qr.displayMatrix(R);
    cout << endl;

    // Verifying Q is orthogonal (Q * Q^T should be identity)
    cout << "Verification - Q * Q^T (should be identity):" << endl;
    qr.displayMatrix(qr.multiplyMatrices(Q, qr.getTranspose(Q)));
    cout << endl;

    // Verifying decomposition (Q * R should equal original matrix)
    cout << "Verification - Q * R (should equal original matrix):" << endl;
    qr.displayMatrix(qr.multiplyMatrices(Q, R));

    // Getting the eigen values and eigen vectors

    auto [eigenValues, eigenVectors] = qr.getEigenValuesandEigenVectors(matrix);

    qr.displayEigenValuesandEigenVectors(eigenValues, eigenVectors);

    return 0;
}
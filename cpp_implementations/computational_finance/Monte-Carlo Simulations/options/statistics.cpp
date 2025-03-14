/**
 * @file statistics.cpp
 * @brief A script to calculate the statistical properties like mean, variance,
 * and covariance.
 *
 * @author Harsh Parikh
 */

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

/**
 * @brief This function calculates the mean of a vector
 *
 * @param num_vector The vector of scalers.
 *
 * @returns The mean of the vector
 */

template <typename T>
double calculateMean(const vector<T> &num_vector)
{

    // This variable stores the average of the input vector
    double average = 0.0;

    // This variable stores the size of the input vector
    double n = num_vector.size();

    for (const auto &element : num_vector)
    {
        average += element / n;
    }
    return average;
}

/**
 * @brief This function calculates the variance of a vector
 *
 * @param num_vector The vector of scalers.
 *
 * @returns The variance of the vector
 */
template <typename T>
double calculateVariance(const vector<T> &num_vector,
                         const bool isPopulation = true)
{

    // Calculating the mean of the vector.
    auto mean = calculateMean(num_vector);

    // Storing the variance
    double variance = 0.0;

    // Evaluating the number of elements
    double n = num_vector.size();

    for (const auto &element : num_vector)
    {
        variance += pow((element - mean), 2);
    }

    // Dividing the variance based on the `isPopulation` parameter
    variance = isPopulation ? (variance / n) : (variance / (n - 1));

    return variance;
}
/**
 * @brief This function calculates the covariance of two vectors
 *
 * @param vec1 A vector of scalers.
 * @param vec2 A vector of scalers.

 * @returns The covariance of the vector
 */
template <typename T>
double calculateCovariance(const vector<T> &vec1, const vector<T> &vec2)
{
    // Check if the dimensions of the vectors are equal
    if (vec1.size() != vec2.size())
    {
        throw invalid_argument("Vectors must be of the same size");
    }

    // Calculating the means of the vectors
    auto mean_vec1 = calculateMean(vec1); // Mean of vector 1
    auto mean_vec2 = calculateMean(vec2); // Mean of vector 2

    // Calculate the covariance
    double covariance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        covariance += (vec1[i] - mean_vec1) * (vec2[i] - mean_vec2);
    }
    covariance /= vec1.size(); // Divide by the number of elements

    return covariance;
}

int main()
{

    // Test variable for calculating average.
    vector<int> nums = {1, 2, 3, 4, 5};

    auto average = calculateMean(nums);

    cout << average << endl;
}
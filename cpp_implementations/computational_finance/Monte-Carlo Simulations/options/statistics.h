#ifndef STATISTICS_H
#define STATISTICS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept> // For invalid_argument

using namespace std;

/**
 * @brief This function calculates the mean of a vector
 *
 * @param num_vector The vector of scalars.
 * @returns The mean of the vector
 */
template <typename T>
double calculateMean(const vector<T> &num_vector)
{
    if (num_vector.empty())
    {
        throw invalid_argument("Cannot compute mean of an empty vector");
    }

    double sum = 0.0;
    for (const auto &element : num_vector)
    {
        sum += element;
    }
    return sum / num_vector.size();
}

/**
 * @brief This function calculates the variance of a vector
 *
 * @param num_vector The vector of scalars.
 * @param isPopulation Whether to calculate population variance (default: true)
 * @returns The variance of the vector
 */
template <typename T>
double calculateVariance(const vector<T> &num_vector, bool isPopulation = true)
{
    if (num_vector.size() < 2 && !isPopulation)
    {
        throw invalid_argument("Sample variance is undefined for n = 1");
    }

    double mean = calculateMean(num_vector);
    double variance = 0.0;
    for (const auto &element : num_vector)
    {
        variance += pow((element - mean), 2);
    }

    return variance / (isPopulation ? num_vector.size() : (num_vector.size() - 1));
}

/**
 * @brief This function calculates the covariance of two vectors
 *
 * @param vec1 A vector of scalars.
 * @param vec2 A vector of scalars.
 * @param isPopulation Whether to calculate population covariance (default: true)
 * @returns The covariance of the vectors
 */
template <typename T>
double calculateCovariance(const vector<T> &vec1, const vector<T> &vec2, bool isPopulation = true)
{
    if (vec1.size() != vec2.size())
    {
        throw invalid_argument("Vectors must be of the same size");
    }
    if (vec1.size() < 2 && !isPopulation)
    {
        throw invalid_argument("Sample covariance is undefined for n = 1");
    }

    double mean_vec1 = calculateMean(vec1);
    double mean_vec2 = calculateMean(vec2);

    double covariance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        covariance += (vec1[i] - mean_vec1) * (vec2[i] - mean_vec2);
    }

    covariance /= (isPopulation ? vec1.size() : (vec1.size() - 1));

    return covariance;
}

#endif // STATISTICS_H

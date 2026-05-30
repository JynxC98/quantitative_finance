/**
 * @brief This header file stores random number generators
 *
 * @author Harsh Parikh
 */

#if !defined(RANDOM_UTILS_HPP)
#define RANDOM_UTILS_HPP

#include <random>

inline unsigned int seed = 42;
inline std::mt19937 gen(seed);
inline std::normal_distribution<double> normal(0.0, 1.0);
inline std::uniform_real_distribution<double> uniform(0.0, 1.0);

#endif
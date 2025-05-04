/**
 * @file auto_diff.cpp
 * @brief Implements forward-mode automatic differentiation using dual numbers.
 *
 * This file defines a Dual number class that supports arithmetic operations
 * enabling automatic computation of derivatives for functions of a single variable.
 * Useful for applications in optimization, sensitivity analysis, and algorithmic differentiation.
 *
 * @author Harsh Parikh
 * @date 4th May 2025
 */
#include <iostream>
#include <cmath>
using namespace std;

/**
 * @class Dual
 * @brief Represents a dual number for forward-mode automatic differentiation.
 *
 * A dual number is of the form (real + ε * dual), where ε is an infinitesimal
 * such that ε² = 0. The `real` part stores the function value and
 * the `dual` part stores the derivative. This class supports arithmetic
 * operations between dual numbers, enabling derivative propagation.
 *
 * @ref https://www.youtube.com/watch?v=QwFLA5TrviI
 */
class Dual
{
private:
    long double real; ///< Real part of the dual number (function value)
    long double dual; ///< Dual part of the dual number (first derivative)

public:
    /**
     * @brief Constructs a Dual number with specified real and dual parts.
     * @param real The real part representing the function value.
     * @param dual The dual part representing the derivative value.
     */
    Dual(long double real, long double dual)
    {
        this->real = real;
        this->dual = dual;
    }

    /**
     * @brief Adds two dual numbers using operator overloading.
     * @param other Another Dual number.
     * @return Result of addition as a Dual number.
     */
    Dual operator+(const Dual &other)
    {
        return Dual(real + other.real, dual + other.dual);
    }

    /**
     * @brief Subtracts one dual number from another using operator overloading.
     * @param other Another Dual number.
     * @return Result of subtraction as a Dual number.
     */
    Dual operator-(const Dual &other)
    {
        return Dual(real - other.real, dual - other.dual);
    }

    /**
     * @brief Multiplies two dual numbers using operator overloading.
     * @param other Another Dual number.
     * @return Result of multiplication as a Dual number.
     *
     * Formula: (a + εb) * (c + εd) = ac + ε(ad + bc)
     */
    Dual operator*(const Dual &other)
    {
        return Dual(real * other.real, real * other.dual + dual * other.real);
    }

    /**
     * @brief Divides one dual number by another using operator overloading.
     * @param other Another Dual number.
     * @return Result of division as a Dual number.
     *
     * Formula: (a + εb) / (c + εd) = (a/c) + ε((bc - ad)/c²)
     */
    Dual operator/(const Dual &other)
    {
        auto new_real = real / other.real;
        auto new_dual = (dual * other.real - other.dual * real) / (other.real * other.real);
        return Dual(new_real, new_dual);
    }
    /**
     * @brief This function returns the value of F(x)
     */
    long double value() { return real; }

    /**
     * @brief This function returns the value of F'(x)
     */
    long double derivative() { return dual; }
};

int main()
{
    // Let F(x) = x^2 + 2x + 3
    // -> F'(x) = 2x + 2

    // Initialising the dual number instance
    Dual x(0.0, 1); // Need to calculate the value of F'(x) at x=0, assuming b = 1

    // In this implementation, it is important to represent all constants as Dual numbers
    Dual y = x * x + Dual(2.0, 0) * x + Dual(3.0, 0);

    cout << "F(X) = " << y.value() << endl;
    cout << "F'(X) = " << y.derivative() << endl;

    return 0;
}

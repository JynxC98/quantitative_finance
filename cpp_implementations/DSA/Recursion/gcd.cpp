/**
 * @file gcd.cpp
 * @brief A script to calculate gcd of two numbers using a recursive function.
 */

#include <iostream>
using namespace std;

int calculateGCD(int a, int b)
{
    if (b == 0) // Base case: when the second number becomes 0
    {
        return a;
    }
    return calculateGCD(b, a % b); // Recursive call using the Euclidean algorithm
}

int main()
{
    int num1 = 5;
    int num2 = 10;

    cout << calculateGCD(num1, num2) << endl;
    return 0;
}

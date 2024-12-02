/**
 * @file demo.cpp
 * @brief A script to demonstrate recursion using CPP
 */

#include <iostream>
using namespace std;

void recursiveFunction(int n)
{
    if (n > 0) // Base case to stop recursion
    {
        cout << n << endl;        // Print the current value
        recursiveFunction(n - 1); // Recursive call
    }
    // When n <= 0, the function stops and returns automatically
}

int main()
{
    int x = 4;
    recursiveFunction(x);
    return 0;
}

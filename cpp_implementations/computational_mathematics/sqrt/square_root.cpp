#include <iostream>

using namespace std;

double square_root(double n){
    /* 
    Newton-Rhapson method to calculate the squareroot of a function.

    f(x) = x^2 - n
    f'(x) = 2x

    new_guess = current_guess - x^2 - n / 2x
    
     */
    
    double root = 0;
    double current_guess = n;
    double tolerance = 1e-8;

    while (true){
        root = current_guess - (pow(current_guess, 2)  - n)/(2 * current_guess);

        if (abs(root - current_guess) < tolerance){
            break;
        }
        current_guess = root; 
    }
    return root;
}

int main(){

    double number = 625.0;

    cout << square_root(number) << endl;

    return 0; 
}
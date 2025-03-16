#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <vector>

using namespace std;

vector<double> thomasAlgorithm(const vector<double> &a,
                               const vector<double> &b,
                               const vector<double> &c,
                               const vector<double> &d,
                               double epsilon = 1e-8);
#endif
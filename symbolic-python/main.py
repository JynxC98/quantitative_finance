"""
A script to solve equations for the expected number of tosses to get HHH
"""

from sympy import symbols, Eq, solve

# Define the variables for the expected values
E0, E1, E2 = symbols("E0 E1 E2")

# Equations based on the state transitions for HHH
eq1 = Eq(E0, 1 + 0.5 * E0 + 0.5 * E1)  # From State 0
eq2 = Eq(E1, 1 + 0.5 * E0 + 0.5 * E2)  # From State 1
eq3 = Eq(E2, 1 + 0.5 * E0)  # From State 2 (E3 = 0)

# Solve the system of equations
solutions = solve((eq1, eq2, eq3), (E0, E1, E2))
print(solutions)

"""
A script for polynomial multiplication using numpy's fft library.

Author: Harsh Parikh
Date: 30th July 2025
"""

import numpy as np


def naive_multiplication(poly_A, poly_B):
    """
    The brute force method of polynomial multiplication.

    Input Parameters
    ----------------
     poly_A : np.ndarray
        Coefficients of the first polynomial, in increasing order of powers (a_0, a_1, ..., a_n).
    poly_B : np.ndarray
        Coefficients of the second polynomial, in increasing order of powers (b_0, b_1, ..., b_m).

    Returns
    -------
    result: np.ndarray
        Coefficients of the product polynomial C(x), of length n + m - 1.
    """
    # Calculating the resultant size
    result_size = len(poly_A) + len(poly_B) - 1

    # Initialising the resultant vector
    result = np.zeros((result_size))

    # Multiplying the polynomials using the naive multiplication method
    for i in range(len(poly_A)):
        for j in range(len(poly_B)):
            result[i + j] += poly_A[i] * poly_B[i]

    return result


def next_power_of_two(n):
    """
    Calculates the next power of two after the number `n`.
    """
    result = 1
    while result < n:
        result <<= 1  # This LOC shifts the bit to the left until the power of two
        # is greater than the input number.
    return result


def multiply_polynomials(poly_A, poly_B):
    """
    Multiply two real-coefficient polynomials using the Fast Fourier Transform (FFT).

    This function takes two 1D NumPy arrays representing the coefficients of polynomials
    A(x) and B(x), and computes their product C(x) = A(x) * B(x) efficiently using
    the convolution theorem and FFT.

    Input Parameters
    ----------
    poly_A : np.ndarray
        Coefficients of the first polynomial, in increasing order of powers (a_0, a_1, ..., a_n).
    poly_B : np.ndarray
        Coefficients of the second polynomial, in increasing order of powers (b_0, b_1, ..., b_m).

    Returns
    -------
    result : np.ndarray
        Real-valued coefficients of the product polynomial C(x), of length n + m - 1.
    """
    # Calculating the total size of the output vector
    result_size = (
        len(poly_A) + len(poly_B) - 1
    )  # -1 accounts for indexing starting at 0.

    # Calculating the next power of two (the size of fft vector must be a power of 2)
    N = next_power_of_two(result_size)

    # Converting the vectors into complex datatype
    A_complex = np.zeros((N), dtype=np.complex64)
    B_complex = np.zeros((N), dtype=np.complex64)

    # Assigning the original vectors to their complex counterparts
    A_complex[0 : len(poly_A)] = poly_A
    B_complex[0 : len(poly_B)] = poly_B

    # Converting the vectors into their forward FFT representation
    A_fft = np.fft.fft(A_complex)
    B_fft = np.fft.fft(B_complex)

    # Multiplying the vectors point-wise
    C_fft = A_fft * B_fft

    # Calculating the inverse of the product of two vectors
    result = np.fft.ifft(C_fft).real

    # The vector is only returned till result size to avoid `0`s
    return result[:result_size]


if __name__ == "__main__":
    vector_A = np.array([1, 1, 1])
    vector_B = np.array([1, 1, 1])

    result = multiply_polynomials(vector_A, vector_B)
    print(result)

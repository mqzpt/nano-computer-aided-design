# Matthew Athanasopoulos - 20976490
import numpy as np
from scipy.linalg import lu

# Coefficient matrix A and right-hand side vector b
# Made diagonally dominant

A = np.array([[-8, 1, -2],
              [2, -6, -1],
              [-3, -1, 7]])

b = np.array([-20, -38, -34])

# Part i) See Attached Photo
# Gauss-Seidel Method: It is an iterative method that can be slower than direct methods like LU decomposition or linalg.solve. Convergence may depend on how well the system is conditioned. However, it can be useful when dealing with large systems where matrix inversion is computationally expensive.

# Part ii) LU Decomposition
P, L, U = lu(A)
y = np.linalg.solve(L, np.dot(P, b))
x_lu = np.linalg.solve(U, y)
print(f'Solution using LU decomposition: {x_lu}')
# LU Decomposition: This method breaks the matrix into lower and upper triangular matrices, making it efficient for solving multiple systems with the same coefficient matrix. It is more stable than inverting the matrix directly but may require more memory.

# Part iii) linalg.solve method
x_solve = np.linalg.solve(A, b)
print(f'Solution using linalg.solve: {x_solve}')
# linalg.solve: This method is efficient and uses optimized algorithms to directly solve the system. It’s the most straightforward and often the fastest option for small to medium-sized systems.

# Part iv) Inverse of A
A_inv = np.linalg.inv(A)
x_inv = np.dot(A_inv, b)
print(f'Solution using inverse of A: {x_inv}')
# Inverse of A: Inverting a matrix is computationally expensive and generally not recommended for solving linear systems due to potential numerical instability. However, it's useful for theoretical purposes or in certain cases where you need the inverse for multiple calculations.
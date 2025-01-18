# Matthew Athanasopoulos - 20976490

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcdefaults()

# Defining the number of nodes and boudaries (r)
dr, r_start, r_end = 0.2, 0, 1
r_nodes = int((r_end - r_start)/dr + 1)
r_vals = np.linspace(r_start, r_end, r_nodes)
n = r_nodes - 1  # Internal nodes

# Define the Right Boundary
u_right, u_init = 1, 0

# Defining the number of nodes and boudaries (t)
dt, t_start, t_end = 0.1, 0, 1.0
t_nodes = int((t_end - t_start)/dt + 1)
t_vals = np.linspace(t_start, t_end, t_nodes)

# Defining Lambda
lam = dt/(2*dr**2)

# Construct the U Matrix with the initial and boundary conditions
U = np.zeros((t_nodes, r_nodes))
U[0, 1:-1] = u_init
U[:, -1] = u_right

# Construct the vector for U_i+1 (U_1) and apply the Derivative Boundary Conidition
U_1 = np.ones(n - 1)
U_1[1:] *= -lam*(1 + dr/(2*r_vals[1:-2]))
U_1[0] = -2*lam

# Construct the vector for U_i-1 (U_N1)
U_N1 = np.ones(n - 1) * -lam*(1 - dr/(2*r_vals[1:-1]))

# Construct the A Matrix with our 3 components
A = np.diag(np.ones(n) * (1 + 2*lam)) + np.diag(U_1, 1) + np.diag(U_N1, -1)

# Initialize first row of A matrix with forward difference
A[0, :] = 0  # Zero out the row
A[0, 0] = 3
A[0, 1] = -4
A[0, 2] = 1

# Perform LU decomposition on matrix A
LU, PIV = lu_factor(A)

# Iterate over each time step to compute the solution
for l in range(t_nodes - 1):

    # Initialize the vector B for the current time step
    B = np.zeros(n)

    # Apply boundary conditions
    B[0] = -U[l, 2] + 4 * U[l, 1] - 3 * U[l, 0]  # Left boundary condition
    B[-1] = (
        2 * lam * (1 + dr / (2 * r_vals[-2])) * u_right
        + (1 - 2 * lam) * U[l, -1]
        + lam * (1 - dr / (2 * r_vals[-2])) * U[l, -2]
    )  # Right boundary condition

    # Populate the remaining elements of B
    for i in range(1, r_nodes - 2):
        ri = r_vals[i]  # Current r value
        B[i] = (
            lam * (1 + dr / (2 * ri)) * U[l, i + 1]
            + (1 - 2 * lam) * U[l, i]
            + lam * (1 - dr / (2 * ri)) * U[l, i - 1]
        )  # Equation for inner nodes

    # Solve the system using LU decomposition and update the U matrix
    U[l + 1, :-1] = lu_solve((LU, PIV), B)

# Plot the final distribution
plt.plot(r_vals, U[-1, :], "-", label=f"Time = {t_vals[-1]:.1f} s")
plt.legend()
plt.xlabel("Radius of Rod (m)")
plt.ylabel("Heat")
plt.title("Heat Distribution in the Rod (CN Method)")
plt.show()

# Start a new figure
plt.figure()

# Select 10 evenly spaced time indices
indices = np.linspace(1, t_nodes - 2, 10, dtype=int)

# Plot intermediate time distributions
for i in indices:
    plt.plot(r_vals, U[i, :], "--", label=f"Time = {t_vals[i]:.1f} s")

# Plot the final time distribution
plt.plot(r_vals, U[-1, :], "-", label=f"Time = {t_vals[-1]:.1f} s")

# Show the plot
plt.legend()
plt.xlabel("Radius of Rod (m)")
plt.ylabel("Heat")
plt.title("Heat Distribution in the Rod (CN Method)")
plt.show()

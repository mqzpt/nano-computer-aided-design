# Matthew Athanasopoulos - 20976490
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Define constants and parameters
L = 1.0  # Length of the pore in cm
CA0 = 1e-3  # Initial concentration of species A at x=0 in mol/cm^3
DA = 1e-3  # Diffusion coefficient of species A in cm^2/s
k = 10  # Reaction rate constant in cm^3/(mol*s)
N = 100  # Number of points
dx = L / (N - 1)  # Step size

# Initialize concentration array with an initial guess (e.g., CA0 throughout)
CA_guess = np.ones(N) * CA0

# Function to set up the system of equations for finite difference solution


def finite_difference_system(CA):
    equations = np.zeros(N)

    # Boundary condition at x=0
    equations[0] = CA[0] - CA0

    # Finite difference equations for interior points
    for i in range(1, N - 1):
        equations[i] = (CA[i+1] - 2*CA[i] + CA[i-1]) - \
            (dx**2 / DA) * k * CA[i]**2

    # Neumann boundary condition at x=L (dCA/dx = 0)
    equations[-1] = CA[-1] - CA[-2]

    return equations


# Solve the system of nonlinear equations
CA_solution = fsolve(finite_difference_system, CA_guess)

# Plot the FD solution
x_vals = np.linspace(0, L, N)
plt.plot(x_vals, CA_solution, label="FD solution", color="green")

# Plot configuration
plt.xlabel("Position (x) [cm]")
plt.ylabel("Concentration of A (C_A) [mol/cm³]")
plt.title("Concentration Profile of Species A Using Finite Difference Method")
plt.legend()
plt.show()

# Convergence in the FD solution can be checked by increasing N and observing if CA(x) stabilizes.
# Additionally, the fsolve function should ideally return residuals close to zero, indicating system consistency.

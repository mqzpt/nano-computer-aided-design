# Matthew Athanasopoulos - 20976490
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve

# Constants and parameters
pore_length, end_x = 1.0, 1.0  # Length of the cylindrical pore in cm
start_x = 0.0  # Starting point of the pore
# Initial concentration of species A at the pore entrance (mol/cm^3)
initial_concentration = 1e-3
# Diffusion coefficient of species A in the pore (cm^2/s)
diffusion_coeff = 1.0e-3
# Reaction rate constant for the reaction A -> B (cm^3/mol*s)
reaction_rate = 10

# Define the differential equations representing concentration and its gradient
# Here, concentration_gradient calculates both dC_A/dx and d^2C_A/dx^2


def concentration_gradient(position, conc):
    concentration, gradient = conc  # conc contains [C_A, dC_A/dx]
    d2c_dx2 = (reaction_rate / diffusion_coeff) * \
        (concentration ** 2)  # d^2C_A/dx^2 = (k/D_A) * C_A^2
    return gradient, d2c_dx2  # Return dC_A/dx and d^2C_A/dx^2

# Define the boundary conditions for the problem
# At x = 0: C_A(x=0) = C_A0 (initial concentration)
# At x = L: dC_A/dx = 0 (no flux boundary condition at closed end)


def boundary_conditions(left, right):
    left_bc = left[0] - initial_concentration  # Enforce C_A(x=0) = C_A0
    right_bc = right[1]  # Enforce dC_A/dx at x=L to be zero
    return left_bc, right_bc

# Function to find the initial guess for the concentration gradient at x=0
# This is used for the shooting method to find an accurate initial slope that satisfies the boundary conditions


def initial_gradient(guess):
    # Solve ODE with the guessed initial gradient at x=0
    solution = solve_ivp(fun=concentration_gradient, t_span=[start_x, end_x],
                         y0=[initial_concentration, guess[0]], method='Radau')
    # Return the error in the boundary condition at x=L for this guess
    return solution.y[1, -1]


# Part (a): Shooting method to find correct initial gradient
# Initial guess for dC_A/dx at x=0 (can be adjusted for better convergence)
initial_guess = [-1e-3]
# Use fsolve to find correct gradient
optimal_gradient = fsolve(initial_gradient, initial_guess)
# Set initial concentration and gradient
initial_conditions = [initial_concentration, optimal_gradient[0]]

# Solve the ODE with solve_ivp using the corrected initial gradient
shooting_solution = solve_ivp(fun=concentration_gradient, t_span=(start_x, end_x),
                              y0=initial_conditions, method='Radau')

# Part (b): Solving the boundary value problem (BVP) directly with solve_bvp
x_points = np.linspace(start_x, end_x, 100)  # Discretize the domain
# Initial guess array for solve_bvp
initial_bvp_guess = np.zeros((2, x_points.size))

# Solve the BVP using solve_bvp
bvp_solution = solve_bvp(fun=concentration_gradient, bc=boundary_conditions,
                         x=x_points, y=initial_bvp_guess)

# Plotting the solutions for comparison
plt.plot(bvp_solution.x, bvp_solution.y[0], label="solve_bvp solution")
plt.plot(shooting_solution.t,
         shooting_solution.y[0], label="solve_ivp solution")
plt.xlabel("Position (x) [cm]")
plt.ylabel("Concentration of A (C_A) [mol/cm³]")
plt.title("Concentration Profile of Species A Along a Cylindrical Pore")
plt.legend()
plt.show()

# Matthew Athanasopoulos - 20976490

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcdefaults()

# True Solution (analytical solution)


def t_true(x): return -(15/2) * (x ** 2) + (82.5 * x) + 75


# Defining the number of nodes and boundary temperatures
n, t_start, t_end,  = 3, 75, 150

# Define X
x_start, x_end = 0, 10
x_vals = np.linspace(x_start, x_end, n + 2)  # Spacing on number of nodes

# Creating the A Matrix and b vector
A = np.diag(np.ones(n) * 0.8) + np.diag(np.ones(n-1) * -0.4, 1) + \
    np.diag(np.ones(n-1) * -0.4, -1)
b = np.ones(n) * 37.5

# Applying the Boundary Conditions
b[0] += t_start * 0.4
b[-1] += t_end * 0.4

# # Solve the System of Equations and add to the initial and final temperatures (and print solution)
sol = np.hstack([t_start, np.linalg.solve(A, b), t_end])
print(sol)

# Make a bunch of points for the analytical solution
x_vals_real = np.linspace(x_start, x_end, 1000)

# Plot the Comparison
plt.plot(x_vals_real, t_true(x_vals_real), label="True Solution")
plt.plot(x_vals, sol, label="FEA Solution")
plt.title("Comparison of True Solution and FEA Solution")
plt.xlabel("Distance (m)")
plt.ylabel("Temperature (Celsius)")
plt.legend()
plt.show()

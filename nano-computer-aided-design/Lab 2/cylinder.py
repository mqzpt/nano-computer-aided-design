# Matthew Athanasopoulos - #20976490

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of ODEs (dT/dr and du/dr)
def ode_system(r, f):
    T, u = f
    dTdr = u
    dudr = -u / r
    return [dTdr, dudr]

# Function to solve IVP for a given guess of u(5)
def solve_ivp_with_guess(u_guess):
    # Initial conditions: T(r=5) = 120, u(r=5) = u_guess
    y0 = [120, u_guess]
    r_vals = np.linspace(5, 10, 100)
    
    # Solve the ODEs using solve_ivp
    sol = solve_ivp(ode_system, [5, 10], y0, t_eval=r_vals)
    return sol

# Shooting method to iteratively adjust u(5) to satisfy T(10) = 60
def shooting_method():
    lower_guess = -5.0  # Lower Bound for u(5)
    upper_guess = 5.0   # Upper Bound for u(5)
    tolerance = 1e-6
    
    for i in range(100):  # Limit to 100 iterations
        u_guess = (lower_guess + upper_guess) / 2  # Midpoint guess
        sol = solve_ivp_with_guess(u_guess)
        
        # Temperature at r=10
        T_at_10 = sol.y[0][-1]
        
        # Check if boundary condition satisfied
        if abs(T_at_10 - 60) < tolerance:
            print(f"Solution found after {i+1} iterations with u(5) = {u_guess}")
            return sol  # Return the solution
        
        # Adjust guess based on the result
        if T_at_10 > 60:
            upper_guess = u_guess  # Lower guess
        else:
            lower_guess = u_guess  # Raise guess
    
    raise RuntimeError("Shooting method did not converge")

# Solve the BVP using shooting method
solution = shooting_method()

# Plot the temperature distribution
plt.plot(solution.t, solution.y[0], label="Temperature (T)")
plt.xlabel('Radius (r)')
plt.ylabel('Temperature (T)')
plt.title('Temperature Distribution using Shooting Method')
plt.legend()
plt.grid(True)
plt.show()

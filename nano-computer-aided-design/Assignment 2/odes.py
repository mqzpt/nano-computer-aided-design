# Matthew Athanasopoulos - 20976490
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def dydt(t, y):
    return (1 + 4 * t) * np.sqrt(y)

# Analytical solution (50 points for a smooth curve)
def analytical_solution(t):
    return ((2*t**2 + t + 2) / 2) ** 2

# Initialize Boundaries and Step Size (h)
ti, tf, h = 0, 1, 0.25

# Get Number of Points
n = int((tf - ti)/h) + 1

# Get T value Ranges
t_values = np.linspace(ti, tf, n)
t_analytical = np.linspace(ti, tf, 50)

# Get the True Solution
y_analytical = analytical_solution(t_analytical)

# Initial Boundary Condition
y0 = 1

# Euler's Method
def euler_method(f, t_values, y0, h):
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])
    return y_values

y_euler = euler_method(dydt, t_values, y0, h)

# Heun's Method
def heuns_method(f, t_values, y0, h):
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        # Take one iteration of Euler Method to get a prediction
        y_predicted = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])
    
        # Correct the value using the average of the two slopes
        y_values[i] = y_values[i-1] + (h / 2) * (f(t_values[i-1], y_values[i-1]) + f(t_values[i-1] + h, y_predicted))
    
    return y_values


y_heun = heuns_method(dydt, t_values, y0, h)

# Ralston's Method
def ralstons_method(f, t_values, y0, h):
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        k1 = f(t_values[i-1], y_values[i-1])
        k2 = f(t_values[i-1] + (2/3) * h, y_values[i-1] + (2/3) * h * k1)
        y_values[i] = y_values[i-1] + (h / 4) * (k1 + 3 * k2)
    return y_values

y_ralston = ralstons_method(dydt, t_values, y0, h)

# Runge-Kutta 4th Order
def rk4_method(f, t_values, y0, h):
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        k1 = f(t_values[i-1], y_values[i-1])
        k2 = f(t_values[i-1] + h / 2, y_values[i-1] + h / 2 * k1)
        k3 = f(t_values[i-1] + h / 2, y_values[i-1] + h / 2 * k2)
        k4 = f(t_values[i-1] + h, y_values[i-1] + h * k3)
        y_values[i] = y_values[i-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_values

y_rk4 = rk4_method(dydt, t_values, y0, h)

# solve_ivp Solver (RK45)
result = solve_ivp(dydt, [ti, tf], [y0], t_eval=t_values, method='RK45')
y_ivp = result.y[0]

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot analytical solution
plt.plot(t_analytical, y_analytical, label="Analytical Solution", color='black', linewidth=2)

# Plot numerical methods as markers
plt.plot(t_values, y_euler, 'o', label="Euler's Method", markersize=8)
plt.plot(t_values, y_heun, 's', label="Heun's Method", markersize=8)
plt.plot(t_values, y_ralston, '^', label="Ralston's Method", markersize=8)
plt.plot(t_values, y_rk4, 'D', label="Runge-Kutta 4th Order", markersize=8)
plt.plot(t_values, y_ivp, 'x', label="solve_ivp Solver (RK45)", markersize=8)

# Labels and legend
plt.title('Comparison of Numerical Methods for Solving ODE')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

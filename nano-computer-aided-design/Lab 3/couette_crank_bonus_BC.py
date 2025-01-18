# Matthew Athanasopoulos - #20976490

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

# Constants and Parameters
v = 0.2
height = 0.1
n = 3
dy = height / (n + 1)
dt = 0.01
p = 994.60
u = 8.931e-4
lam = (u * dt) / (2 * p * (dy ** 2))

# Stability Check
if lam > 0.25:
    print("Oscillatory behavior might occur due to instability")

# Boundary conditions
v_initial = 0
v_lower = 0.2

# Spatial and temporal grids
y_points = np.linspace(0, height, n + 2)
time_end = 25000
time_steps = int(time_end / dt) + 1
time_points = np.linspace(0, time_end, time_steps)

# Initialize velocity array
v_array = np.zeros((time_steps, n + 2))
v_array[0, 1:-1] = v_initial
v_array[:, 0] = v_lower

# Create tridiagonal matrix for LU decomposition
main_diag = (1 + 2 * lam) * np.ones(n)
upper_diag = (-lam) * np.ones(n - 1)
lower_diag = (-lam) * np.ones(n - 1)
matrix = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)

# LU decomposition
lu, piv = lu_factor(matrix)

# Time-stepping loop
steady_state_reached = False
steady_state_time = None
tolerance = 1e-8
window_size = 100

for t in range(time_steps - 1):
    rhs = (
        lam * v_array[t, :-2]
        + (1 - 2 * lam) * v_array[t, 1:-1]
        + lam * v_array[t, 2:]
    )
    rhs[0] += lam * v_array[t, 0]

    # Apply Neumann boundary condition for the top boundary
    # Top boundary gradient condition: v[n+1] = v[n]
    rhs[-1] += lam * v_array[t, -2]

    # Solve the system
    v_array[t + 1, 1:-1] = lu_solve((lu, piv), rhs)

    # Enforce Neumann BC explicitly
    v_array[t + 1, -1] = v_array[t + 1, -2]

    # Check for steady state using a rolling window
    if not steady_state_reached and t >= window_size:
        max_change = np.max(
            np.abs(v_array[t + 1 - window_size:t + 1, 1:-1] -
                   v_array[t - window_size:t, 1:-1])
        )
        if max_change < tolerance:
            steady_state_reached = True
            steady_state_time = t * dt

# Plot velocity profile over y for selected times
plt.figure()
selected_times = [0, time_steps // 2, time_steps - 1]
for t in selected_times:
    plt.plot(y_points, v_array[t, :], label=f'Time = {t * dt:.2f} s')
plt.xlabel('Height (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Profile vs. Height with Neumann BC')
plt.legend()
plt.show()

# Plot velocity evolution over time at selected nodes
plt.figure()
selected_nodes = [1, (n + 1) // 2, n]
for node in selected_nodes:
    plt.plot(time_points, v_array[:, node], label=f'y = {node * dy:.2f} m')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Evolution at Selected Nodes with Neumann BC')
plt.legend()
plt.show()

# Print lambda value
print(f"Lambda value: {lam}")

# Print steady state information
if steady_state_reached:
    print(
        f"System reached steady state at approximately t = {steady_state_time:.2f} seconds.")
else:
    print("System did not reach steady state within the given simulation time.")

# System reached steady state at approximately t = 14903.69 seconds, again this feels like too long...

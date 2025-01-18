# Matthew Athanasopoulos - 20976490
import numpy as np

# Define constants and grid parameters
dx, dy = 0.1, 0.1
left_boundary_temp, bottom_boundary_temp = 50, 20
ambient_temp = 25
sigma = 1e-4
grid_size_x, grid_size_y = 3, 3
node_count = grid_size_x * grid_size_y

# Initialize the temperature grid with boundary conditions for sigma = 1e-4
temperature_grid = np.zeros((4, 4))
temperature_grid[:, 0] = left_boundary_temp
temperature_grid[-1, :] = bottom_boundary_temp

# Iterative solution for sigma = 1e-4 with relaxation

# Iterative parameters
max_iterations = 100
tolerance = 1e-8
relaxation_factor = 1.1
error_matrix = np.ones((4, 4)) * 100
iteration_count = 0

# Perform the relaxation method to solve the non-linear equation iteratively
while iteration_count < max_iterations and np.max(error_matrix[1:3, 1:3]) > tolerance:
    old_temperature_grid = temperature_grid.copy()

    for row in range(grid_size_x - 1, -1, -1):
        for col in range(1, grid_size_y + 1):
            prev = temperature_grid[row, col]

            # Calculate new value based on neighboring points
            if row == 0 and col == grid_size_y:
                temperature_grid[row, col] = 0.25 * (2 * old_temperature_grid[row - 1, col] +
                                                     2 * old_temperature_grid[row, col - 1] + dx**2 * sigma * (ambient_temp - old_temperature_grid[row, col]**4))
            elif row == 0:
                temperature_grid[row, col] = 0.25 * (old_temperature_grid[row - 1, col] + old_temperature_grid[row + 1, col] +
                                                     2 * old_temperature_grid[row, col - 1] + dx**2 * sigma * (ambient_temp - old_temperature_grid[row, col]**4))
            elif col == grid_size_y:
                temperature_grid[row, col] = 0.25 * (2 * old_temperature_grid[row - 1, col] + old_temperature_grid[row, col - 1] +
                                                     old_temperature_grid[row + 1, col] + dx**2 * sigma * (ambient_temp - old_temperature_grid[row, col]**4))
            else:
                temperature_grid[row, col] = 0.25 * (old_temperature_grid[row - 1, col] + old_temperature_grid[row + 1, col] +
                                                     old_temperature_grid[row, col - 1] + old_temperature_grid[row, col + 1] + dx**2 * sigma * (ambient_temp - old_temperature_grid[row, col]**4))

            # Apply relaxation and update the error matrix
            temperature_grid[row, col] = relaxation_factor * \
                temperature_grid[row, col] + (1 - relaxation_factor) * prev
            error_matrix[row, col] = abs(
                (temperature_grid[row, col] - prev) / temperature_grid[row, col])

    iteration_count += 1

# Display the final solution for sigma = 1e-4
print("\nTemperature distribution for sigma = 1e-4:")
print(temperature_grid)

# Define the system of equations and solve for sigma = 0

# Construct the coefficient matrix for the interior nodes
main_diag = -4 * np.ones(node_count)
upper_diag_1 = np.ones(node_count - 1)
lower_diag_1 = np.ones(node_count - 1)
upper_diag_3 = np.ones(node_count - 3)
lower_diag_3 = np.ones(node_count - 3)

# Assemble the matrix using diagonals
A = np.diag(main_diag) + np.diag(upper_diag_1, 1) + np.diag(lower_diag_1, -1) \
    + np.diag(upper_diag_3, 3) + np.diag(lower_diag_3, -3)

# Adjust matrix for boundary conditions
for i in range(3):
    A[6 + i, 3 + i] = 2
    A[2 + (i * grid_size_x), 1 + (i * grid_size_y)] = 2

for i in range(2):
    A[3 + (i * grid_size_x), 2 + (i * grid_size_x)] = 0
    A[2 + (i * grid_size_x), 3 + (i * grid_size_x)] = 0

# Boundary condition vector for sigma = 0
b = np.zeros(node_count)
b[:grid_size_x] -= bottom_boundary_temp
b[::grid_size_y] -= left_boundary_temp

# Solve the linear system to get the internal temperature values
temp_solution_sigma_0 = np.linalg.solve(A, b)

# Initialize the temperature grid with boundary values for sigma = 0
temperature_grid = np.zeros((4, 4))
temperature_grid[:, 0] = left_boundary_temp
temperature_grid[-1, :] = bottom_boundary_temp

# Insert solution back into the temperature grid
temperature_grid[2, 1:] = temp_solution_sigma_0[:grid_size_x]
temperature_grid[1, 1:] = temp_solution_sigma_0[grid_size_x:2 * grid_size_x]
temperature_grid[0, 1:] = temp_solution_sigma_0[2 *
                                                grid_size_x:3 * grid_size_x]

# Display the solution for sigma = 0
print("\nTemperature distribution for sigma = 0:")
print(temperature_grid)

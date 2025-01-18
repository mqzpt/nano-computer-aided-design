# Matthew Athanasopoulos - 20976490
from scipy.integrate import solve_ivp
import time


def ode_system(t, y):
    [x1, x2, x3] = y
    dx1_dt = (-0.013 * x1) - (1000 * x1 * x3)
    dx2_dt = (-2500 * x2 * x3)
    dx3_dt = (-0.013 * x1) - (1000 * x1 * x3) - (2500 * x2 * x3)
    return [dx1_dt, dx2_dt, dx3_dt]


# Loop through Every IVP Method
results = {}
for solver in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]:

    # Solve each system, and record the time by taking the diff between start and end time
    # Initial conditions are [1, 1, 0] as mentioned in the question
    time_start = time.time()
    sol = solve_ivp(ode_system, (0, 50), [1, 1, 0], solver)
    time_end = time.time()
    _time = time_end - time_start

    # Store results so we can print them at the end
    results[solver] = _time

# Print out the Results (time taken for each method)
print("\nAverage Time of each Method:\n")
for solver, _time in results.items():
    print(f"Solver: {solver} Time taken: {_time}s")


# Explicit Methods are limited by stability conditions and require small time steps for stiff systems, leading to potential fluctuations and instability. Implicit Methods are unconditionally stable and allow for larger time steps, resulting in more efficient and accurate solutions for stiff problems. This is a stiff problem due the the large difference in time scales between the fast and slow variables. The fast variables are the x1 and x3, while the slow variable is x2. The fast variables will change rapidly, while the slow variable will change slowly. This difference in time scales makes the problem stiff, and implicit methods are better suited to handle stiff problems. Basically although the implict methods are more computationally expensive (and thus take more time as we see), they provide more accurate solutions for stiff problems.

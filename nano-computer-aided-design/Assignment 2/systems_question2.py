# Matthew Athanasopoulos - 20976490
import numpy as np
from scipy.optimize import fsolve
from time import perf_counter

# Make the system of equations


def equations(vars):
    Tc, Jc, Th, Jh = vars
    eq1 = 5.67e-8 * Tc**4 + 17.41 * Tc - Jc - 5188.18
    eq2 = Jc - 0.71 * Jh + 7.46 * Tc - 2352.71
    eq3 = 5.67e-8 * Th**4 + 1.865 * Th - Jh - 2250
    eq4 = Jh - 0.71 * Jc + 7.46 * Th - 11093
    return [eq1, eq2, eq3, eq4]


# Initial guess
initial_guess = [298, 3000, 298, 5000]

# Part a) Solve using fsolve
start = perf_counter()
solution_fsolve = fsolve(equations, initial_guess)
end = perf_counter()
print("Solution using fsolve:", solution_fsolve)
print("Time taken by fsolve:", end - start)

# Part b) Newton's method with analytical Jacobian


def jacobian(vars):
    Tc, Jc, Th, Jh = vars
    jac = np.zeros((4, 4))
    jac[0, 0] = 4 * 5.67e-8 * Tc**3 + 17.41
    jac[0, 1] = -1
    jac[0, 2] = 0
    jac[0, 3] = 0
    jac[1, 0] = 7.46
    jac[1, 1] = 1
    jac[1, 2] = 0
    jac[1, 3] = -0.71
    jac[2, 0] = 0
    jac[2, 1] = 0
    jac[2, 2] = 4 * 5.67e-8 * Th**3 + 1.865
    jac[2, 3] = -1
    jac[3, 0] = 0
    jac[3, 1] = -0.71
    jac[3, 2] = 7.46
    jac[3, 3] = 1
    return jac


def newtons_method(tol=1e-4):
    x = np.array(initial_guess, dtype=float)
    max_iter = 100
    start = perf_counter()

    for i in range(max_iter):
        f = np.array(equations(x))
        j = jacobian(x)
        delta_x = np.linalg.solve(j, -f)
        x_new = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            break
        x = x_new

    end = perf_counter()
    print("Solution using Newton's method:", x)
    print("Time taken by Newton's method:", end - start)
    print(f"Number of iterations: {i+1}")


newtons_method()

# Part c) Newton's method with numerical Jacobian


def numerical_jacobian(vars, h=1e-6):
    n = len(vars)
    j = np.zeros((n, n))
    f0 = np.array(equations(vars))

    for i in range(n):
        x_forward = np.copy(vars)
        x_backward = np.copy(vars)
        x_forward[i] += h
        x_backward[i] -= h

        f_forward = np.array(equations(x_forward))
        f_backward = np.array(equations(x_backward))

        j[:, i] = (f_forward - f_backward) / (2 * h)

    return j


def newtons_method_numerical(tol=1e-4):
    x = np.array(initial_guess, dtype=float)
    max_iter = 100
    start = perf_counter()

    for i in range(max_iter):
        f = np.array(equations(x))
        j = numerical_jacobian(x)
        delta_x = np.linalg.solve(j, -f)
        x_new = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            break
        x = x_new

    end = perf_counter()
    print("Solution using Newton's method (numerical Jacobian):", x)
    print("Time taken:", end - start)
    print(f"Number of iterations: {i+1}")


newtons_method_numerical()

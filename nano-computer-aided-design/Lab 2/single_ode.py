# Matthew Athanasopoulos - #20976490

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------
# --define const at top--
# -----------------------
x_0 = 10
t_span = [5, 15]

# this is the space to define values such as x(t=5)=10 etc that you will use in all of your code

# ---------------------
# -------Task 1.1------
# ---------------------

# Defining our Beta value
beta = -2

# Defining the ODE function


def ode_fun_1(t, x):
    return beta * x


# Initial Conditions and Time Span
t_eval = np.linspace(5, 15, 100)


# Solving the ode and calling the solution sol1
sol1 = solve_ivp(ode_fun_1, t_span, [x_0], t_eval=t_eval)

# Plot x vs t for all t values with a suitable label
plt.plot(sol1.t, sol1.y[0], label=f'beta = {beta}')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Solution of dx/dt = beta * x with beta = -2')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------
# -------Task 1.2------
# ---------------------
# find the value of x at t=7 by using commands in code (not just reading from graph)

t_eval_7 = np.array([7])
sol1_7 = solve_ivp(ode_fun_1, t_span, [x_0], t_eval=t_eval_7)

# Printing the value of x at t=7
print(f'x(t=7) = {sol1_7.y[0][0]}')

# ---------------------
# -------Task 1.3------
# ---------------------
# write your ode as a function that now takes three arguments with beta being the last one.
# call this ode_fun_2


def ode_fun_2(t, x, beta):
    return beta * x


# Now we're solving for beta = -2 with the args param
sol2 = solve_ivp(ode_fun_2, t_span, [x_0], t_eval=t_eval, args=(-2,))

# plot x vs t for all t values with a suitable label in a new figure
plt.plot(sol2.t, sol2.y[0], label=f'beta = -2')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Solution of dx/dt = beta * x with beta as a parameter')
plt.legend()
plt.grid(True)
plt.show()


# ---------------------
# -------Task 1.4------
# ---------------------

betas = [-2, -4, -6, -8]
for beta in betas:
    sol = solve_ivp(ode_fun_2, t_span, [x_0], t_eval=t_eval, args=(beta,))
    plt.plot(sol.t, sol.y[0], label=f'beta = {beta}')

# Add labels and plot
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Solutions for different values of beta')
plt.legend()
plt.grid(True)
plt.show()

# Bonus: include labels to show the value of beta for each solution being plotted - DONE!

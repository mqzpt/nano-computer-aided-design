# Matthew Athanasopoulos - #20976490

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

#-----------------------
#--define const at top--
#-----------------------
# Initial conditions
# initial rabbit population
r0 = 0.5
# initial fox population
f0 = 0.3

initial_conditions = [r0, f0]

# Time span from 0 to 20
t_span = [0, 20]
t_eval = np.linspace(0, 20, 200)

#---------------------
# Define the system of ODEs
def predator_prey_system(t, z):
    r, f = z  # unpack r and f
    drdt = r * (1 - 2 * f)
    dfdt = -f * (1 - r)
    return [drdt, dfdt]

#---------------------
# Solve the system of ODEs
sol = solve_ivp(predator_prey_system, t_span, initial_conditions, t_eval=t_eval)

# Extract solutions for plotting

# Rabbit population over time
r_values = sol.y[0]  
# Fox population over time
f_values = sol.y[1]  

#---------------------
#-------Task 2.1------
# Plot the results for rabbits and foxes over time
plt.plot(sol.t, r_values, label="Rabbits (r(t))")
plt.plot(sol.t, f_values, label="Foxes (f(t))")
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Model: Population vs Time')
plt.legend()
plt.grid(True)
plt.show()

#---------------------
#-------Task 2.2------
# Create a phase plot (rabbits on the y-axis and foxes on the x-axis)
plt.plot(f_values, r_values)
plt.xlabel('Foxes (f(t))')
plt.ylabel('Rabbits (r(t))')
plt.title('Phase Plot: Foxes vs Rabbits')
plt.grid(True)
plt.show()

#---------------------
#-------Bonus Task------
# Creating an animation of the phase plot building over time
fig, ax = plt.subplots()

# Had to make the plot limits a bit more so it's easier to look at
ax.set_xlim([0, 1.25])
ax.set_ylim([0, 2.25])
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Foxes (f(t))')
ax.set_ylabel('Rabbits (r(t))')
ax.set_title('Phase Plot Animation: Rabbits vs Foxes')

# Init function
def init():
    line.set_data([], [])
    return line,

# Animation function updates the plot frame-by-frame
def animate(i):
    line.set_data(f_values[:i], r_values[:i])
    return line,

# Create the animation (this is in stdlib)
ani = animation.FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, blit=True, interval=100)
plt.show()

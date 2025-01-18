from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

C0=0.03   #mol/L
D = 1.5e-9 * 1e6**2   #m2/s coverted to micron2/s
k = 35   #s-1
L = 50   # micron


def myode(x,y):
	# y = [C u]
	dydx=[y[1],k*y[0]/D]
	return dydx

def resid(u0):
	sol = solve_ivp(myode,[0,L],[C0,u0[0]])
	r = np.abs(sol.y[1,-1])+np.abs(sol.y[0,-1])
	return r

ucorrect=fsolve(resid,-10)

u_correct=ucorrect[0]

# now get the solution 
sol = solve_ivp(myode,[0,L],[C0,u_correct])


plt.figure()
plt.plot(sol.t,sol.y[0],label='C')
plt.plot(sol.t,sol.y[1],label='dCdx')
plt.legend()
plt.xlabel('Depth [micron]')
plt.show()



from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def dT_func(t,T,dx):
	dT=np.zeros(T.size) 
	a = 0.081 
	T0 = 10 - 14*np.cos(2*np.pi*(t-37)/365) 
	dx2=dx**2 
	dT[0] = a * (T0-2*T[0]+T[1])/dx2 
	dT[1:-1] = a * (T[:-2]-2*T[1:-1]+T[2:])/dx2 
	dT[-1] = a * (2*T[-2]-2*T[-1])/dx2
	return dT

# set up the grid
# to check for convergence, ideally we should try different 
# values of dx and max depth 
dx=2 #can select this
(x0,xend)=(0,20)
n=int((xend-x0)/dx + 1) #number of points in x

x=np.linspace(x0,xend,n) # includes surface (where T is known)


xnodes = x[1:] # internal nodes only
               # there will be one ode per node
			    
#solve the problem
tspan=[0,4*365]     # four years
T0=10*np.ones(xnodes.size)   # need to select some initial conditions
                             # we don't know these, which is why
                             # we should solve for several years
sol=solve_ivp(lambda t,T:dT_func(t,T,dx),tspan,T0)   # solve the ode
T0 = 10 - 14*np.cos(2*np.pi*(sol.t-37)/365)    # calculate surface T
                                           # this is also hard coded in the ode
                                           # (not ideal coding to have it twice)
Temps=np.vstack([T0,sol.y]) #all the Ts

# present the results
# first, a regular plot of the T at each node. 
# the sinusoidal oscillation at the surface decays in magnitude
# and has increasing lag as depth increases
plt.figure()
for i in range(n):
	plt.plot(sol.t/365,Temps[i,:],label='node'+str(i)) 
plt.xlabel('Time [d]')
plt.ylabel('T [degC]')
plt.legend()
plt.show()

plt.figure()
for i in range(0,sol.t.size):
	plt.cla()
	plt.ylim([-5,20])
	plt.xlim([0,20])
	plt.plot(x,Temps[:,i])
	plt.title('Temperature profile at t = '+str(int(sol.t[i]))+'days')
	plt.xlabel('Depth [m]')
	plt.ylabel('T [degC]')
	plt.pause(0.1)
plt.show()

#you could also try 3d plots as we have done in class
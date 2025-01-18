from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

C0=0.03   #mol/L
D = 1.5e-9 * 1e6**2   #m2/s coverted to micron2/s
k = 35   #s-1
L = 50   # micron
n = 9 
dx = L/(n+1)   # micron


def dcdt_de(_,C):
	# c = [c1 c2 c3 ... cn+1]
	# initialise dcdt
	dcdt = np.zeros(C.size)  
	Ddx2 = D/dx**2 #since we need this a lot 
	# first interior node
	dcdt[0] = Ddx2*(C[1]-2*C[0]+C0)- k*C[0]  
	# work out the values for other interior nodes
	dcdt[1:-1] = Ddx2*(C[2:]-2*C[1:-1]+C[:-2])- k*C[1:-1] 
	# and the boundary node
	dcdt[-1] = 2*Ddx2*(C[-2]-C[-1])- k*C[-1]
	return dcdt
	
# now get the solution 
sol = solve_ivp(dcdt_de,[0,0.1],np.zeros(n+1))

#put back BC
conc=np.vstack([C0*np.ones(sol.t.size),sol.y])

x_vals=np.linspace(0,L,n+2)

#plot conc vs depth as time goes on
for i in range(0,sol.t.size):
	plt.plot(x_vals,conc[:,i])
	plt.title('CO2 conc. profile at t = '+str(sol.t[i])+'s')
	plt.xlabel('Depth [micron]')
	plt.ylabel('CO2 Conc. [mol/L]')
plt.show()



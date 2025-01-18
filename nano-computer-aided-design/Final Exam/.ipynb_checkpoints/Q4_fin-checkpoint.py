from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import numpy as np

#parameters
h = 20   #W/m2/K
k = 14   #W/m/K
Tair = 20 
L = 0.1   #m

def dydx(x,y):
	 #y = [T u]
	 # where u = r^2 dT/dx
	 r = lambda x:0.02 - 0.1*x
	 rx=r(x)

	 dydx_arr = [y[1]/rx**2, 2*h/k*rx*(y[0]-Tair)]
	 return dydx_arr

def mybc(y0,yL):
	 # y0 = y(x = 0) = [T(0) u(0)]
	 # yL = y(x = L) = [T(L) u(L)]
	 r = lambda x:0.02 - 0.1*x
	 resid = [y0[0]-100,
              h*(yL[0]-Tair)+k*yL[1]/r(L)**2]
	 return resid    


# solve using solve_bvp  
x_vals=np.linspace(0,L,100)
sol=solve_bvp(dydx,mybc,x_vals,np.zeros((2,x_vals.size)))


#plot results

plt.figure()
plt.plot(sol.x,sol.y[0],'k-',label='T')
plt.plot(sol.x,sol.y[1],'r-',label='dTdx')
plt.legend()
plt.show()
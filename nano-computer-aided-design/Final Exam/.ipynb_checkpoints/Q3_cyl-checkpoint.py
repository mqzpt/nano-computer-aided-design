import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

#parameters
Qdot=1e3 #W
k=14     #W/mK
R1=5e-2  #m
R2=10e-2 #m
BC1=120  #degC


#r grid
n=6 #total number of points in r, includes BCs
r=np.linspace(R1,R2,n)

#internal r
#note we need the radius at the end as well
ri=r[1:]

#need dr for formula 
dr=r[1]-r[0]


A=np.diag(-2*np.ones(n-1))+\
  np.diag(1-dr/(2*ri[1:]),-1)+\
  np.diag(1+dr/(2*ri[:-1]),1) 




#now for b  
b=np.zeros(n-1)
b[0]=(dr/(2*r[1])-1)*BC1
 

#BC at R2
A[-1,-1]=3
A[-1,-2]=-4
A[-1,-3]=1

print(A)

b[-1]=-Qdot*dr/(k*np.pi*R2)


#T array
T=np.zeros(n)
#BC at R1
T[0]=BC1



#solve for unknown 
T[1:]=np.linalg.solve(A,b)
print('The temperature distribution is')
print(T)

#finding the gradients

dTdr=np.zeros(n) 
#lets use a forward difference
#for the first node
#dT/dr=(1/2dr)(-Ti+2+4Ti+1-3Ti)
dTdr[0]=(1/2/dr)*(-3*T[0]+4*T[1]-T[2])

#central for the interior
#dTdr=(Ti+1-Ti-1)/(2 delr)
dTdr[1:-1]=(1/(2*dr))*(T[2:]-T[:-2]) 

#backward diff for last node
#dT/dr=(1/2dr)(+3Ti-4Ti-1+Ti-2)
dTdr[-1]=(1/(2*dr))*(3*T[-1]-4*T[-2]+T[-3]) 

# now check the BC seems ok - should get 0 for this:
check = -k*dTdr[-1]-Qdot/(2*np.pi*R2)

print("this should be zero", check)

# obtain the requested numbers
# T at R2
print('The T(r=R2) =', T[-1])
# dTdr at R1
print('dTdr(r=R1) =', dTdr[0])


#plot results 

fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('r [m]')
ax1.set_ylabel('T [degC] ', color=color)
ax1.plot(r, T, color=color,linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc=4)
	
	
ax2 = ax1.twinx()  

color = 'tab:green'
ax2.set_ylabel('dT/dr', color=color)  
ax2.plot(r,dTdr, color=color,linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc=1)
	
fig.tight_layout() 
plt.show()
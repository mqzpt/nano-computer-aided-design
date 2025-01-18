import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def dxdt(t,x):
	#x=[M S]
	return [10,2-10*x[1]/x[0]]

sol=solve_ivp(dxdt,[0,120],[1000,200],t_eval=np.linspace(0,120,100))


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('t(min)')
ax1.set_ylabel('M ', color=color)
ax1.plot(sol.t,sol.y[0],'r-')
ax1.tick_params(axis='y',labelcolor=color)


ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(sol.t,sol.y[1])
ax2.set_ylabel('S ', color=color)
ax2.tick_params(axis='y',labelcolor=color)


plt.figure()
plt.plot(sol.t,sol.y[1]/sol.y[0])
plt.ylabel('S/M')
plt.title('fraction of salt in the tank')

plt.show()
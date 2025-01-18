from scipy.integrate import solve_ivp,solve_bvp
import matplotlib.pyplot as plt
import numpy as np

# =================== #
# Define parameters   #
# =================== #
# here, we define all the parameters at the top of 
#the file and use them below
L = 0.001  # m
CA0 = 0.2  # mol/L
uL = 0     # mol/(L m) boundary condition
k = 0.001  # 1/s
D = 1.2e-9 # m2/s


def dcdx(x,C):
	# inputs x : position
	#        C = [CA  : concentration of A
	#             u ] : value of dCA/dx
	# outputs dC/dt = [dCA/dt   = [u
	#                  du/dt ]     kCA/D]
	dCdx = [C[1], k*C[0]/D]
	
	return dCdx

    
def mybc(C0,CL):
	# inputs C0 = [CA(0)  u(0)]'
	#        CL = [CA(L)  u(L)]'
	resid = [C0[0]-CA0,CL[1]-uL]
	return resid

# ====================== #
# solve using solve_ivp  #
# ====================== #
guess1 = -CA0/L  # need an initial guess of u(0) - assume the 
                 # concentration drops to 0 in a straight line
guess2 = guess1*2  # need a second guess, since it is linear it won't really matter

# solve the problem with the first guess and find the predicted u(L)
sol1 = solve_ivp(dcdx,[0,L],[CA0,guess1],t_eval=np.linspace(0,L,100)) 
uLg1 = sol1.y[-1,-1] # this is uL for guess gamma0

# and again with the second guess  and find the predicted u(L)
sol2 = solve_ivp(dcdx,[0,L],[CA0,guess2],t_eval=np.linspace(0,L,100)) 
uLg2 = sol2.y[-1,-1]  # this is uL for guess gamma1

# now interpolate or extrapolate to get the correct initial guess
# (note: if this was a non-linear system you would have to iterate)
u0 = guess1 - (uLg1 - uL) * (guess2-guess1)/(uLg2-uLg1) #

# now solve the problem with this correct initial guess
sol = solve_ivp(dcdx,[0,L],[CA0,u0],t_eval=np.linspace(0,L,100)) 


# ====================== #
# solve using solve_bvp  #
# ====================== #

x_vals=np.linspace(0,L,100)
sol_bv=solve_bvp(dcdx,mybc,x_vals,np.zeros((2,x_vals.size)))

# ====================== #
# present some findings  #
# ====================== #

# display some relevant values, 
# here we use print to display them to the screen
print(f'Solution obtained used {u0:.2f} mol/(L m) ')
print(f'Predicted CA(L) {sol.y[0,-1]:.2f} mol/L')


# ====================== #
# plot the results       #
# ====================== #


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('x (m)')
ax1.set_ylabel('cA (M) ', color=color)
ax1.plot(sol.t, sol.y[0,:], color=color,linewidth=2,label='shooting method')
ax1.plot(sol_bv.x,sol_bv.y[0],'k-.',label='solve_bvp')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc=4)
	
	
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('dc/dx', color=color)  # we already handled the x-label with ax1
ax2.plot(sol.t, sol.y[1,:], color=color,linewidth=2,label='shooting method')
ax2.plot(sol_bv.x,sol_bv.y[1],'k-.',label='solve_bvp')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc=1)
	
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
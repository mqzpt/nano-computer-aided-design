from scipy.integrate import solve_ivp, solve_bvp
import matplotlib.pyplot as plt
import numpy as np


def dxdt(t, x):
    # x = [S, M]
    return [10, 2 - 10*(x[0]/x[1])]


sol = solve_ivp(dxdt, [0, 150], [200, 1000], t_eval=np.linspace(0, 150, 100))

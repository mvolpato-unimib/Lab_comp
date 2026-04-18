import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import Newt_Rap

import lib_plot


z_sp = sp.symbols('z')
sp_fz = (z_sp**2 - 1)*(z_sp**2 + 1)

fz = sp.lambdify(z_sp, sp_fz, 'numpy')
der_fz = sp.lambdify(z_sp, sp.simplify(sp.diff(sp_fz, z_sp)), 'numpy')

N = 1024
x = np.linspace(-1,1,N).astype(np.float64)
X, Y = np.meshgrid(x, x)
z_init = X + 1j*Y

c_mat = Newt_Rap(fz, der_fz, z_init, MaxIter=100)

c_angle = np.angle(abs(c_mat))
print(c_angle)

fig = plt.figure()
plt.imshow(abs(np.angle(c_mat)), cmap="viridis")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import Newt_Rap

fz = lambda z: z**4 - 1
der_fz = lambda z: 4 * z**3
N = 3000
x = np.linspace(-1,1,N).astype(np.float64)
X, Y = np.meshgrid(x, x)
z_init = X + 1j*Y

c_mat = Newt_Rap(fz, der_fz, z_init, MaxIter=200, der_eps=1e-5)

np.savetxt('fract.txt', c_mat)
print('Output saved on file: "fract.txt"')

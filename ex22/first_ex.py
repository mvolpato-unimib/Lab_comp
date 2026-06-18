import numpy as np
import math

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import Gauss_Herm, root_finder, H_weigts, herm_coeff

func = lambda x: 1 / (1+x**2)

deg = 15
xcoo = root_finder(herm_coeff(deg))
ycoo = H_weigts(xcoo, deg)

res = Gauss_Herm(func, deg)
print(f'Result (n={deg}) = ', res)

name = 'txts/res_test_gauss.txt'
data_to_save = np.column_stack((xcoo, ycoo))
np.savetxt(name, data_to_save, fmt='%.8f')
with open(name, 'a') as f:
    f.write(f"\n{res} None")

print(f"\nData saved in {name}\n")


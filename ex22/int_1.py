import numpy as np
import math

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import Gauss_Herm

def func(x):
    return 1 / (1+x**2)

n_deg_1 = np.arange(1, 23, 2)
true_1 = np.e * np.pi * math.erfc(1)

def main():
    data = []
    print('Start computations:')
    for deg in n_deg_1:
        res = Gauss_Herm(func, deg)
        err = abs(res - true_1)
        data.append([deg, res, err])
    print('END')

    data = np.array(data)
    np.savetxt('txts/int1.txt', data, fmt='%d %.18e %.1e')

    return 0

if __name__ == '__main__':
    main()

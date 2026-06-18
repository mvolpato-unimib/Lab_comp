import numpy as np
import math

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import Gauss_Herm

def func(x): 
    return x**5 * np.where(x<3, np.zeros_like(x), np.ones_like(x))

true_3 = 101*np.exp(-9) / 2 
n_deg_3 = np.arange(5, 23, 2)

def main():
    data = []
    print('Start computations:')
    for deg in n_deg_3:
        res = Gauss_Herm(func, deg)
        err = abs(res - true_3)
        data.append([deg, res, err])
    print('END')

    data = np.array(data)
    np.savetxt('txts/int3.txt', data, fmt='%d %.18e %.1e')
    return 0

if __name__ == '__main__':
    main()

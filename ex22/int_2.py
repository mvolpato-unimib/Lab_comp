import numpy as np
import math

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import Gauss_Herm

def func(x): 
    return x**8 /2

n_deg_2 = np.arange(1, 23, 2)
true_2 = 105/32 * np.sqrt(np.pi)

def main():
    data = []
    print('Start computations:')
    for deg in n_deg_2:
        res = Gauss_Herm(func, deg)
        err = abs(res - true_2)
        data.append([deg, res, err])
    print('END')

    data = np.array(data)
    np.savetxt('txts/int2.txt', data, fmt='%d %.18e %.1e')
    return 0

if __name__ == '__main__':
    main()

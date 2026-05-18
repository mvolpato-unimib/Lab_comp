
import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import bisec, trapezoidal, Simpson
import numpy as np
import time


diff_I_t = lambda x: trapezoidal(func, x, x + dx_, dx_)
diff_I_s = lambda x: Simpson(func, x, x + 2*dx_, dx_)

folder = 'out\\'

def main():
    xcoo = np.linspace(a+dx_, x_max, 100)


    # computations

    print('\nComputing the Integral...')
    t0 = time.perf_counter()
    int_s = np.array([I_s(x) for x in xcoo])
    t_int_s = time.perf_counter() - t0
    print(f"integral.txt: {t_int_s:.4f} s")


    print('\nComputing ycoo_t and ycoo_s...')
    t1 = time.perf_counter()
    ycoo_t = np.abs(np.array([diff_I_t(x) for x in xcoo]))
    t_ycoo_t = time.perf_counter() - t1
    print(f"ycoo_t.txt:   {t_ycoo_t:.4f} s")


    t2 = time.perf_counter()
    ycoo_s = np.abs(np.array([diff_I_s(x) for x in xcoo]))
    t_ycoo_s = time.perf_counter() - t2
    print(f"ycoo_s.txt:   {t_ycoo_s:.4f} s")


    print('\nComputing b_t and b_s...')
    t3 = time.perf_counter()
    b_t = bisec(lambda x: diff_I_t(x) - eps, 30, 50, tol=eps/10)
    t_b_t = time.perf_counter() - t3
    print(f"x_max_t.txt:  {t_b_t:.4f} s")


    t4 = time.perf_counter()
    b_s = bisec(lambda x: diff_I_s(x) - eps, 30, 50, tol=eps/10)
    t_b_s = time.perf_counter() - t4
    print(f"x_max_s.txt:  {t_b_s:.4f} s")



    # print on file

    np.savetxt(folder + 'integral.txt', int_s)
    np.savetxt(folder + 'ycoo_t.txt', ycoo_t)
    np.savetxt(folder + 'ycoo_s.txt', ycoo_s)
    np.savetxt(folder + 'x_max.txt', [b_t, b_s])

    return 0

if __name__ == '__main__':
    main()
    
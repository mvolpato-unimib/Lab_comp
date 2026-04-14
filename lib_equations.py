import numpy as np
import matplotlib.pyplot as plt

def Part_dervs(func, params, eps=1e-8):
    part_der = []
    for i, p in enumerate(params):
        diff_pars = params.copy()
        diff_pars[i] += eps
        part_der.append((func(*diff_pars) - func(*params))/eps)
    return part_der


def bisec(f, a, b, optim=True, tol=1e-14, out_niter=False):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError('f(a) and f(b) must have different signs')

    n_iter = 0
    while True:
        n_iter += 1
        
        if optim:
            den = fb - fa
            if abs(den) < tol: 
                c = (a + b) / 2
            else:
                c = (a * fb - b * fa) / den
        else:
            c = (a + b) / 2

        fc = f(c)
        
        if abs(fc) < tol or abs(b - a) < tol:
            break

        if fa * fc > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc

    return (c, n_iter) if out_niter else c



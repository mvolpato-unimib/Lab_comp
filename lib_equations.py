import numpy as np
import matplotlib.pyplot as plt

def Part_dervs(func, params, eps=1e-8):
    part_der = []
    for i, p in enumerate(params):
        diff_pars = params.copy()
        diff_pars[i] += eps
        part_der.append((func(*diff_pars) - func(*params))/eps)
    return part_der


def bisec(f, a, b, optim=True, tol=1e-14, out_niter=False, out_story=False):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError('f(a) and f(b) must have different signs')

    story = []
    n_iter = 0
    while True:
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
        
        story.append(c)
        n_iter += 1

    if out_niter: 
        return (c, n_iter) 
    elif out_story:
        return (c, n_iter, story) 
    else:
        return c


def Newt_Rap(func, der_func, x0, tol=1e-14, MaxIter=1e2):
    xn = x0
    f_val = func(xn)
    f_der_val = der_func(xn)
    x1 = x0 - f_val / f_der_val    
    n_iter = 0

    sol_ls = []

    while abs(func(x1)) > tol and n_iter<MaxIter:
        f_val = func(x1)
        f_der_val = der_func(x1)
        x_next = lambda x_old: x_old - f_val / f_der_val    
        
        if f_der_val == 0:
            raise ValueError("Derivative is 0. Newton would break.")
        x0 = x1
        x1 = x_next(x0)
        sol_ls.append(x1)
        n_iter += 1
    return x1, n_iter, np.array(sol_ls)



def Secant_mth(func, x0, x1, tol=1e-14, MaxIter=1e2):
    der_func = lambda x, x_prev: (func(x) - func(x_prev)) / (x - x_prev)
    n_iter = 0
    x_prev = x0
    x = x1
    f_val = func(x)
    f_der_val = der_func(x, x_prev)
    x_next = x - f_val / f_der_val
    sol_ls = []

    while abs(f_val) > tol and n_iter < MaxIter:
        fx_next = lambda x_old: x_old - f_val / f_der_val    
        if f_der_val == 0:
            raise ValueError("Derivative is 0. Secant mth would break.")
        x_prev = x
        x = fx_next(x)
        sol_ls.append(x)
        n_iter += 1
        f_val = func(x)
        f_der_val = der_func(x, x_prev)
    return x, n_iter, np.array(sol_ls)



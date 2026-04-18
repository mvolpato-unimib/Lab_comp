import numpy as np
import matplotlib.pyplot as plt

def Part_dervs(func, params, eps=1e-8):
    part_der = []
    for i, p in enumerate(params):
        diff_pars = params.copy()
        diff_pars[i] += eps
        part_der.append((func(*diff_pars) - func(*params))/eps)
    return part_der


def bisec(f, a, b, optim=True, tol=1e-14, out_niter=False, out_story=False, MaxIter=500):
    a_og, b_og = [a, b]
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError('\nf(a) and f(b) must have different signs')

    story = []
    n_iter = 0
    while n_iter < MaxIter:
        if optim:
            den = fb - fa
            if abs(den) < tol: 
                c = (a + b) / 2
            else:
                c = (a * fb - b * fa) / den
        else:
            c = (a + b) / 2

        fc = f(c)
        

        if fa * fc > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc
        
        story.append(c)
        n_iter += 1
        
        if abs(fc) < tol or abs(b - a) < tol:
            break
    
    if n_iter >= MaxIter:
        import warnings
        warnings.warn(f'\nMaximum iterations reached for a = {a_og:.2f}, b ={b_og:.2f}', RuntimeWarning)
    

    if out_niter: 
        return (c, n_iter) 
    elif out_story:
        return (c, n_iter, np.array(story)) 
    else:
        return c


def Newt_Rap(func, der_func, x0, tol=1e-14, MaxIter=500):
    from collections.abc import Iterable
    
    # if input is iterable, list, tuple, np array, ecc.
    if isinstance(x0, Iterable): 
        if not isinstance(x0, np.ndarray):
            xn = np.array(x0)
        else:
            xn = x0
        n_iter = np.zeros_like(x0, dtype=int)  

        truth_cond = np.ones_like(x0, dtype=bool)
        while np.all(n_iter < MaxIter):

            f_val = np.where(truth_cond, func(xn), 0)
            truth_cond = abs(f_val) > tol
            
            f_der_val = np.where(truth_cond, der_func(xn), 0)
            n_iter += np.where(truth_cond, 1, 0)
            xn = np.where(truth_cond, xn - f_val / f_der_val, xn)
            
            if not np.any(truth_cond):
                break
            
            # print('Iter.', np.max(n_iter))

        if np.max(n_iter) >= MaxIter:
            import warnings
            warnings.warn(f'\nMaximum iterations reached. Algorithm diverges!', RuntimeWarning)
        return xn

    # if input is a scalar
    else:
        xn = x0
        sol_ls = [x0]
        n_iter = 0

        while n_iter < MaxIter:
            f_val = func(xn)
            f_der_val = der_func(xn)

            if abs(f_val) < tol:
                break

            if abs(f_der_val) < 1e-20:
                import warnings
                warnings.warn(f'\nDerivative too small at x = {xn:.2f}', RuntimeWarning)
                break

            xn = xn - f_val / f_der_val
            sol_ls.append(xn)
            n_iter += 1

            if np.isnan(f_val):
                break

        if n_iter >= MaxIter:
            import warnings
            warnings.warn(f'\nMaximum iterations reached for x0 = {x0:.2f}', RuntimeWarning)
        
        if np.isnan(func(xn)):
            import warnings
            warnings.warn(f'\nNaN detected at iteration {n_iter} for x0 = {x0:.2f}', RuntimeWarning)
        
        return xn, len(sol_ls), np.array(sol_ls)



def Secant_mth(func, x0, x1, tol=1e-14, MaxIter=500):
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
            raise ValueError("\nDerivative is 0. Secant mth would break.")
        x_prev = x
        x = fx_next(x)
        sol_ls.append(x)
        n_iter += 1
        f_val = func(x)
        f_der_val = der_func(x, x_prev)

    if n_iter >= MaxIter:
        import warnings
        warnings.warn(f'\nMaximum iterations reached for x0 = {x0:.2f}, x1={x1:.2f}', RuntimeWarning)
    return x, n_iter, np.array(sol_ls)



import numpy as np
import matplotlib.pyplot as plt

class DFT:
    def __init__(self, x_desc):
        """Initializes the Discrete Fourier Transform (DFT) object.

        Args:
            x_desc (array-like): A 1D array of spatial or temporal coordinates.
        """

        N = len(x_desc)
        L = (x_desc[-1] - x_desc[0])
        
        if N % 2 == 0: 
            p_pos = np.arange(N//2+1)  # positive values bcs of Niquist (even)
            p_neg = np.arange(1, N - (N//2)) * (-1)
        else: 
            p_pos = np.arange(N//2)  # positive values bcs of Niquist (odd)
            p_neg = np.arange(1, N - (N//2)) * (-1)

        p = 2 * np.pi / L * np.concatenate((p_pos, p_neg))
        W = np.exp(1j * np.outer(p, x_desc))

        self.W = W
        self.p = p
        self.N = N
        self.L = L
        self.dx = L / N
    
    def dft(self, y_desc):
        """Computes the Discrete Fourier Transform.

        Args:
            y_desc (array-like): A 1D array of signal values corresponding to the initialized coordinates.
        """

        if type(y_desc)==int or len(y_desc) != self.N:
            raise ValueError('len(y_desc) must be equal to len(x_desc)!!')
        y_dft = self.dx * self.W @ y_desc
        return y_dft

    def idft(self, y_desc):
        """Computes the Inverse Discrete Fourier Transform.

        Args:
            y_desc (array-like): A 1D array of frequency domain values to invert.
        """

        if type(y_desc)==int or len(y_desc) != self.N:
            raise ValueError('len(y_desc) must be equal to len(x_desc)!!')
        y_idft = self.L**(-1) * self.W.conj().T @ y_desc
        return y_idft




# ----------------------------------------------------------
# ROOT FINDERS
# ----------------------------------------------------------


def bisec(f, a, b, optim=True, tol=1e-8, out_niter=False, out_story=False, MaxIter=500):
    """Finds a root of a function within a specified interval using the bisection method.

    Args:
        f (callable): The continuous function for which to find a root.
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        optim (bool, optional): If True, uses a secant-like optimization for the midpoint. Defaults to True.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-8.
        out_niter (bool, optional): If True, returns the number of iterations performed alongside the root. Defaults to False.
        out_story (bool, optional): If True, returns the history of root approximations. Defaults to False.
        MaxIter (int, optional): The maximum allowed iterations before termination. Defaults to 500.
    """

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




def Newt_Rap(func, der_func, x0, tol=1e-14, MaxIter=500, der_eps=1e-14):
    """Finds a root of a function using the Newton-Raphson method.

    Args:
        func (callable): The target function for which to find a root.
        der_func (callable): The analytical derivative of the target function.
        x0 (float or array-like): The initial guess(es) for the root.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-14.
        MaxIter (int, optional): The maximum allowed iterations. Defaults to 500.
        der_eps (float, optional): The added epsilon to compensate the zero division error on the derivative. Default to 1e-14.
    """

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
            
            f_der_val = np.where(truth_cond, der_func(xn + der_eps), 0)
            n_iter += np.where(truth_cond, 1, 0)
            xn = np.where(truth_cond, xn - f_val / f_der_val, xn)
            
            if not np.any(truth_cond):
                break
            
            # print('Iter.', np.max(n_iter))

        if np.max(n_iter) >= MaxIter:
            import warnings
            warnings.warn(f'\n\nMaximum iterations reached. Algorithm diverges!', RuntimeWarning)
        return xn

    # if input is a scalar
    else:
        xn = x0
        sol_ls = [x0]
        n_iter = 0

        while n_iter < MaxIter:
            f_val = func(xn)
            f_der_val = der_func(xn + der_eps)

            if abs(f_val) < tol:
                break

            if abs(f_der_val) < 1e-20:
                import warnings
                warnings.warn(f'\n\nDerivative too small at x = {xn:.2f}', RuntimeWarning)
                break

            xn = xn - f_val / f_der_val
            sol_ls.append(xn)
            n_iter += 1

            if np.isnan(f_val):
                break

        if n_iter >= MaxIter:
            import warnings
            warnings.warn(f'\n\nMaximum iterations reached for x0 = {x0:.2f}', RuntimeWarning)
        
        if np.isnan(func(xn)):
            import warnings
            warnings.warn(f'\n\nNaN detected at iteration {n_iter} for x0 = {x0:.2f}', RuntimeWarning)
        
        return xn, len(sol_ls), np.array(sol_ls)




def Newt_Rap_corr(func, der_func, x0, tol=1e-14, MaxIter=500, der_eps=1e-14, lamb_shift=0.85):
    """Finds a root of a function using the Newton-Raphson method, modified to reduce cycles introducing a correction factor.

    Args:
        func (callable): The target function for which to find a root.
        der_func (callable): The analytical derivative of the target function.
        x0 (float or array-like): The initial guess(es) for the root.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-14.
        MaxIter (int, optional): The maximum allowed iterations. Defaults to 500.
        der_eps (float, optional): The added epsilon to compensate the zero division error on the derivative. Default to 1e-14.
        lamb_shift (float, optional): represent the shift from the original algorithm, introduced to reduce some phenomena like cycles. Must be in (0,1). Defualt to 0.85.
    """

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
            
            f_der_val = np.where(truth_cond, der_func(xn + der_eps), 0)
            n_iter += np.where(truth_cond, 1, 0)
            xn = np.where(truth_cond, xn - lamb_shift * f_val / f_der_val, xn)
            
            if not np.any(truth_cond):
                break
            
            # print('Iter.', np.max(n_iter))

        if np.max(n_iter) >= MaxIter:
            import warnings
            warnings.warn(f'\n\nMaximum iterations reached. Algorithm diverges!', RuntimeWarning)
        return xn

    # if input is a scalar
    else:
        xn = x0
        sol_ls = [x0]
        n_iter = 0

        while n_iter < MaxIter:
            f_val = func(xn)
            f_der_val = der_func(xn + der_eps)

            if abs(f_val) < tol:
                break

            if abs(f_der_val) < 1e-20:
                import warnings
                warnings.warn(f'\n\nDerivative too small at x = {xn:.2f}', RuntimeWarning)
                break

            xn = xn - lamb_shift * f_val / f_der_val
            sol_ls.append(xn)
            n_iter += 1

            if np.isnan(f_val):
                break

        if n_iter >= MaxIter:
            import warnings
            warnings.warn(f'\n\nMaximum iterations reached for x0 = {x0:.2f}', RuntimeWarning)
        
        if np.isnan(func(xn)):
            import warnings
            warnings.warn(f'\n\nNaN detected at iteration {n_iter} for x0 = {x0:.2f}', RuntimeWarning)
        
        return xn, len(sol_ls), np.array(sol_ls)





def Secant_mth(func, x0, x1, tol=1e-14, MaxIter=500):
    """Finds a root of a function using the secant method.

    Args:
        func (callable): The target function for which to find a root.
        x0 (float): The first initial guess.
        x1 (float): The second initial guess.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-14.
        MaxIter (int, optional): The maximum allowed iterations. Defaults to 500.
    """

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





def root_finder(cn, rand_shift=True, shift=1, nmax=1e3):
    """Finds the roots of a polynomial by computing the eigenvalues of its companion matrix.

    Args:
        cn (array-like): The polynomial coefficients, ordered from highest to lowest degree.
        rand_shift (bool, optional): If True, applies a random shift to improve eigensolver stability. Defaults to True.
        shift (float, optional): A fixed shift to apply if rand_shift is False. Defaults to 1.
        nmax (int or float, optional): The maximum number of iterations for the QR eigensolver. Defaults to 1e3.
    """

    from lib_algebra import QR_eigensolver  
    # the algorithm works with a vector of params ordered from c0, to cn 
    # where P(x) = c0 + c1*x + ... + cn*x^n 
    # [cn, ... , c0] ----> [c0, ... , cn]   
    cn = np.flip(cn)            

    cn_bar = cn / cn[-1]
    diag_mat = np.eye(len(cn)-2, len(cn)-1, k=1) 
    min_coeff = (-1) * cn_bar[:-1]
    comp_mat = np.vstack((diag_mat, min_coeff))

    eigens = None

    if rand_shift:
        while eigens is None:
            current_shift = np.random.rand()   
            shifted_mat = comp_mat + current_shift * np.eye(len(comp_mat))
            try:
                results = QR_eigensolver(shifted_mat, N_max=nmax)
                eigens = np.real(results[0] - current_shift)
            except:
                pass
    else:    
        shifted_mat = comp_mat + shift * np.eye(len(comp_mat))
        eigens = np.real(QR_eigensolver(shifted_mat, N_max=nmax)[0] - shift)
    
    return np.sort(eigens)

# ----------------------------------------------------------
# END ROOT FINDERS
# ----------------------------------------------------------




# ----------------------------------------------------------
# HERMITE COEFFICIENTS
# ----------------------------------------------------------
def herm_coeff(n):
    """Evaluate the Hermite polynomials coefficients of degree n.

    Args:
        n (int): Degree of the polynomial.
    """
    if n == 0:
        return np.array([1])
    if n == 1:
        return np.array([2, 0])
    
    h_minus_1 = np.array([1])   # H0
    h_n = np.array([0, 2])      # H1
    
    for i in range(1, n):
        term1 = 2 * np.insert(h_n, 0, 0) 
        term2 = 2 * i * np.append(h_minus_1, [0, 0])
        h_next = term1 - term2
        h_minus_1 = h_n
        h_n = h_next
        
    return np.flip(h_n)


def herm_func(x, p_deg): 
    """Evaluate the Hermite polynomials of degree p in a range x

    Args:
        x (float or array-like): Points where to evaluate the polynomial.
        p_deg (int): Degree of the polynomial.
    """
    
    coeffs = np.flip(herm_coeff(p_deg))
    monomial = x[:, None]**np.arange(0, p_deg+1)
    res_mat = monomial * coeffs

    return np.sum(res_mat, axis=1)
# ----------------------------------------------------------
# END HERMITE COEFFICIENTS
# ----------------------------------------------------------




# ----------------------------------------------------------
# RESOLUTION OF DIFFERENTIAL EQUATIONS
# ----------------------------------------------------------



def Part_dervs(func, params, eps=1e-8):
    """Calculates the partial derivatives of a function using finite differences.

    Args:
        func (callable): The target function to differentiate.
        params (list or array-like): The sequence of parameters at which to evaluate the derivatives.
        eps (float, optional): The step size for the finite difference approximation. Defaults to 1e-8.
    """

    part_der = []
    for i, p in enumerate(params):
        diff_pars = params.copy()
        diff_pars[i] += eps
        part_der.append((func(*diff_pars) - func(*params))/eps)
    return part_der




def euler(f_xy, y0, x):
    """Solves an ordinary differential equation using the explicit Euler method.

    Args:
        f_xy (callable): The derivative function f(x, y) defining the ODE.
        y0 (float or array-like): The initial condition(s) at the starting point.
        x (array-like): The grid of independent variable points at which to evaluate the solution.
    """

    leny = len(y0) if not np.isscalar(y0) else 1
    sols_y = np.zeros((len(x), leny))
    sols_y[0] = y0
    for i in range(len(x) - 1):
        h = x[i+1] - x[i]
        sols_y[i+1] = sols_y[i] + h * f_xy(x[i], sols_y[i])
    return sols_y




def back_euler(f_xy, y0, x):
    """Solves an ordinary differential equation using the backward (implicit) Euler method.

    Args:
        f_xy (callable): The derivative function f(x, y) defining the ODE.
        y0 (float or array-like): The initial condition(s) at the starting point.
        x (array-like): The grid of independent variable points at which to evaluate the solution.
    """

    from lib_equations import Secant_mth
    leny = len(y0) if not np.isscalar(y0) else 1
    sols = np.zeros((len(x), leny))
    sols[0] = y0
    for i in range(len(x) - 1):
        h = x[i+1] - x[i]
        y = lambda eta_p1: eta_p1 - sols[i] - h*f_xy(x[i+1], eta_p1)
        
        sol = Secant_mth(y, x[i], x[i+1])
        sols[i+1] = sol[0]
    return sols




def rk2(f_xy, y0, x):
    """Solves an ordinary differential equation using the second-order Runge-Kutta method.

    Args:
        f_xy (callable): The derivative function f(x, y) defining the ODE.
        y0 (float or array-like): The initial condition(s) at the starting point.
        x (array-like): The grid of independent variable points at which to evaluate the solution.
    """

    leny = len(y0) if not np.isscalar(y0) else 1
    sols_y = np.zeros((len(x), leny))
    sols_y[0] = y0
    for i in range(len(x) - 1):
        h = x[i+1] - x[i]
        # RK coefficients
        k1 = f_xy(x[i], sols_y[i])
        k2 = f_xy(x[i] + h/2, sols_y[i] + h* k1 /2)

        sols_y[i+1] = sols_y[i] + h * k2
    return sols_y





def rk4(f_xy, y0, x):
    """Solves an ordinary differential equation using the fourth-order Runge-Kutta method.

    Args:
        f_xy (callable): The derivative function f(x, y) defining the ODE.
        y0 (float or array-like): The initial condition(s) at the starting point.
        x (array-like): The grid of independent variable points at which to evaluate the solution.
    """

    leny = len(y0) if not np.isscalar(y0) else 1
    sols_y = np.zeros((len(x), leny), dtype=np.complex128)
    sols_y[0] = y0
    for i in range(len(x) - 1):
        h = x[i+1] - x[i]
        # RK coefficients
        k1 = f_xy(x[i], sols_y[i])
        k2 = f_xy(x[i] + h/2, sols_y[i] + h* k1 /2)
        k3 = f_xy(x[i] + h/2, sols_y[i] + h* k2 /2)
        k4 = f_xy(x[i+1], sols_y[i] + h* k3)

        phi = 1/6 * (k1 + 2*k2 + 2*k3 +k4)

        sols_y[i+1] = sols_y[i] + h * phi
    return sols_y

# ----------------------------------------------------------
# END RESOLUTION OF DIFFERENTIAL EQUATIONS
# ----------------------------------------------------------




# ----------------------------------------------------------
# NUMERICAL INTEGRATION
# ----------------------------------------------------------
def loc_trapezoidal(f, x0, dx=1e-5):
    """Evaluate the integral of a given function f, on a given interval (x0, x0+dx), using local Trapezoidal rule algorithm.

    Args:
        f (function): The function onto the integral is evaluated.
        x0 (float): Lower extreme of the interval
        dx (float, optional): Step for the evaluation in the algorithm. Default to 1e-5. 
    """
    return dx/2 * (f(x0) + f(x0+dx))



def loc_Simpson(f, x0, dx=1e-5):
    """Evaluate the integral of a given function f, on a given interval (x0, x0+dx), using local Simpson's rule algorithm.

    Args:
        f (function): The function onto the integral is evaluated.
        x0 (float): Lower extreme of the interval
        dx (float, optional): Step for the evaluation in the algorithm. Default to 1e-5. 
    """
    return dx/6 * (f(x0) + 4*f(x0 + dx/2) + f(x0+dx))





def trapezoidal(f,a, b, dx=1e-5):
    """Evaluate the integral of a given function f, on a given interval (a, b), using Trapezoidal rule algorithm.

    Args:
        f (function): The function onto the integral is evaluated.
        a (float): Lower extreme of the interval
        b (float): Upper extreme of the interval
        dx (float, optional): Step for the evaluation in the algorithm. Default to 1e-5. 
    """
    Dx = np.abs(b-a)
    N = int(Dx / dx)
    print(N)
    f_arr = f(a + np.arange(1, N)*dx)
    return dx/2 * (f(a) + 2*np.sum(f_arr) + f(b))



def Simpson(f,a, b, dx=1e-5):
    """Evaluate the integral of a given function f, on a given interval (a, b), using Simpson's rule algorithm.

    Args:
        f (function): The function onto the integral is evaluated.
        a (float): Lower extreme of the interval
        b (float): Upper extreme of the interval
        dx (float, optional): Step for the evaluation in the algorithm. Default to 1e-5. 
    """
    Dx = np.abs(b-a)
    N_try = int(Dx / dx)
    N = N_try if N_try%2==0 else N_try+1
    dx = Dx / N

    f_arr = f(a + np.arange(1, N)*dx)
    f_even = f_arr[1::2]
    f_odd = f_arr[0::2]
    
    return dx/3 * (f(a) + 2*np.sum(f_even) + 4*np.sum(f_odd) + f(b))
   



def Gauss_Leg(f,a, b, dx=1e-5):
    """Evaluate the integral of a given function f, on a given interval (a, b), using 2-points Gauss Legendre algorithm.

    Args:
        f (function): The function onto the integral is evaluated.
        a (float): Lower extreme of the interval
        b (float): Upper extreme of the interval
        dx (float, optional): Step for the evaluation in the algorithm. Default to 1e-5. 
    """
    Dx = np.abs(b-a)
    N = int(Dx / dx)
    dx = Dx / N
    c_p, c_m = ((1 + 1/np.sqrt(3))/2,(1 - 1/np.sqrt(3))/2)
    arr_p = dx * (np.arange(N) + c_p)
    arr_m = dx * (np.arange(N) + c_m)
    
    return dx/2 * np.sum(f(a+arr_p) + f(a+arr_m))
   

def H_weigts(x, n): 
    from math import factorial
    num = 2**(n-1) * factorial(n-1) * np.sqrt(np.pi)
    den = n * (herm_func(x, n-1))**2
    return num / den


def Gauss_Herm(f, n=10):
    """Evaluates the integral of a function f(x) * e^(-x^2) using n-point Gauss-Hermite quadrature.
    
    This algorithm approximates the value of the integral over the entire real line:
    ∫ f(x) * e^(-x^2) dx ≈ Σ [w_i * f(x_i)] , (i in [1, n])

    Args:
        f (callable): The function f(x) to be integrated.
        n (int, optional): Number of sample points and weights (degree of the polynomial). Defaults to 10.

    """

    x_i = root_finder(herm_coeff(n))
    return np.sum(H_weigts(x_i, n) * f(x_i))




# ----------------------------------------------------------
# END NUMERICAL INTEGRATION
# ----------------------------------------------------------

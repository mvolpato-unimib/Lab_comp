import numpy as np
import lib_algebra

class Lagrange:
    """
    Object that enables Lagrange interpolation on a given set of points

    :param x: Coordinates x of the points
    :param y: Coordinates y of the points
    """
    def __init__(self, x, f):
        self.x = x
        self.f = f
        n = len(x)
        w_arr = []
        for k in range(n):
            mask = np.ones(len(x), dtype=bool)
            mask[k] = False
            wk = np.prod((x[k] - x[mask] + 1e-15))**(-1)
            w_arr.append(wk)
        self.w = np.array(w_arr)

    def __call__(self, x0):
        frac = self.w / (x0 - self.x)
        return np.sum(self.f * frac) / np.sum(frac)
    

def lagrange_int(x, f, N_points=1000):
    """
    Function that extract the points from the interpolation of f, useful for drawing with plt 
    """
    lag = Lagrange(x, f)
    xoff = 0.001*(max(x) - min(x))
    x_in = np.linspace(x[0]-xoff, x[-1]+xoff, N_points)
    y_in = []
    for i in x_in:
        y_in.append(lag(i))
    return x_in, np.array(y_in)

def Cheby_nodes(dim):
    j = np.arange(dim)
    return -np.cos(j*np.pi / (dim-1))



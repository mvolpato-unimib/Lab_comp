import numpy as np
import matplotlib.pyplot as plt
import lib_plot

def Direct_mth(xarr, yarr, N_points=1000):
    from lib_algebra import QR_solver

    ls = []
    for i in range(len(xarr)):
        ls.append(xarr**i)
    vanderm =  np.array(ls).T
    a_arr = np.flip(QR_solver(vanderm, yarr))

    xoff = 0.001*(max(xarr) - min(xarr))
    x_in = np.linspace(xarr[0]-xoff, xarr[-1]+xoff, N_points)
    y_in = np.polyval(a_arr, x_in)
    return x_in, y_in

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


def fit_engine(x, y, covar, poly_order):
    from scipy.stats import chi2
    from lib_algebra import mat_inv, BackChol
    
    linfunc_ls = []
    for i in np.flip(np.arange(poly_order)+1):
        f_i = lambda x, i=i: x**i
        linfunc_ls.append(f_i)
    linfunc_ls.append(lambda x: np.ones_like(x))
    linfunc = np.array(linfunc_ls)

    X_ls = [f(x) for f in linfunc]
    X = np.array(X_ls).T
    W = mat_inv(covar)
    M = X.conj().T @ W @ X
    b = X.T @ W @ y
    
    # Parameters of the fit
    pars = np.real(BackChol(M, b))
    cov_par = mat_inv(M)
    
    r = y - X @ pars
    Chi2 = np.real(r.T @ W @ r)
    dof = len(x) - len(linfunc)
    print('dof =', dof)
    print()
    p = chi2.sf(Chi2, dof)

    return list(pars), np.real(cov_par), Chi2, p
    

def eval_Par(par_func, old_pars, old_cov_pars):
    from lib_equations import Part_dervs
    J_mat_ls = []
    new_pars = []
    for pf in par_func:
        part_der = Part_dervs(pf, old_pars)
        J_mat_ls.append(part_der)
        new_pars.append(pf(*old_pars))

    J_mat = np.array(J_mat_ls)
    new_cov_pars = J_mat @ old_cov_pars @ J_mat.T
    return new_pars, new_cov_pars


def plot_fit (f, x_sc, y_sc, yerr, params, cov_par,
              start, stop, fit_name='Fit function', nsigma=1,
              xlab='X COO', ylab='Y COO', save_name='plots/plot.png'):
    eps = 1e-8
    x_plot = np.linspace(start, stop, 500)
    y_plot = np.array([f(x, *params) for x in x_plot])
    J_mat_ls = []
    for x in x_plot:
        part_der = []
        for i, p in enumerate(params):
            diff_pars = params.copy()
            diff_pars[i] += eps
            part_der.append((f(x, *diff_pars) - f(x, *params))/eps)
        J_mat_ls.append(part_der)
    
    J_mat = np.array(J_mat_ls)
    sy = nsigma * np.sqrt(np.diag(J_mat @ cov_par @ J_mat.T))    # choice of # sigma to plot

    plt.plot(x_plot, f(x_plot, *params), label=fit_name, color='red')
    plt.errorbar(x_sc, y_sc, yerr=yerr, fmt='o', label='Dati')
    plt.fill_between(x_plot, y_plot+sy, y_plot-sy, alpha=0.2, color='red', label=rf'Errore a ${nsigma}\sigma$')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.savefig(save_name)
    plt.show()





def lin_fit (x, y, cov, poly_order, name_pars=None, 
             mar='.', col='red', name_points='name points',
             labels=[r'x', r'y'], nsigma=3, lims=None, 
             ErrTrue=True, save_name=None, write_formula=False):
    import matplotlib.patches as mpatches

    yerr = np.sqrt(np.diag(cov))
    fit_res = fit_engine(x, y, cov, poly_order)
    pars, cov_pars, chi2, pval = fit_res
    err_pars = np.sqrt(np.diag(cov_pars))
    
    # print on screen results of the fit
    if name_pars != None:
        for i, name in enumerate(name_pars):
            print(f'{name} = {pars[i]:.2e} +/- {err_pars[i]:.3e}')
        print()
        print(f'Chi2 = {chi2:.1f}')
        print(f'p-val = {pval:.2f}')


    # from a list of linear functions: [a, b*x, c*x^2, ...]
    # extract a single lambda func that represent: f(x) = a + b*x + c*x^2 + ...
    def f(x_coo, parameters):
        sol_ls = [np.flip(parameters)[i] * x_coo**i for i in range(len(parameters))]
        return np.sum(np.array(sol_ls), axis=0)

    eps = 1e-14
    start, stop = [min(x), max(x)]
    x_plot = np.linspace(start, stop, 500)
    y_plot = np.array([f(x, pars) for x in x_plot])
    J_mat_ls = []
    for x_ele in x_plot:
        part_der = []
        for i, p in enumerate(pars):
            diff_pars = pars.copy()
            diff_pars[i] += eps
            part_der.append((f(x_ele, diff_pars) - f(x_ele, pars))/eps)
        J_mat_ls.append(part_der)
    
    J_mat = np.array(J_mat_ls)
    sy = nsigma * np.sqrt(np.diag(J_mat @ cov_pars @ J_mat.T))    # choice of n.sigma to plot



    fig, ax = plt.subplots(figsize=(8, 5))
    # plot of:
    #     - Fit 
    #     - Points w/Errorbar
    #     - Error Band

    ax.plot(x_plot, f(x_plot, pars), label='Fit '+ name_points, color=col,
            alpha=0.4, zorder=0, ls='--')
    if ErrTrue:
        ax.errorbar(x, y, yerr, ls='', 
                    marker=mar, color=col, ms=10, label=name_points)
        ax.fill_between(x_plot, y_plot+sy, y_plot-sy, alpha=0.2, color=col, label=rf'Errore a ${nsigma}\sigma$')
    else:
        ax.scatter(x, y, marker=mar,
                    color=col, label=name_points)

    # labels and lims
    if lims!=None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    handles, labels = plt.gca().get_legend_handles_labels()
    
    def add_empty_handle(label_text):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False))
        labels.append(label_text)

    # fit formula
    letters = ['a', 'b', 'c', 'd']

    if write_formula:
        fit_form = r'$f(x) = a'
        for i in range(1, poly_order): 
            if i == 1:
                fit_form += f' + {letters[i]}x'
            else:
                fit_form += f' + {letters[i]}x^{i}'
        fit_form += '$'
        add_empty_handle(fit_form)
    

    for i in range(len(pars)):
        par_lab = rf'${letters[i]} = {pars[i]:.2f} \pm {err_pars[i]:.2f}$'
        add_empty_handle(par_lab)
    
    ax.legend(handles=handles, labels=labels)

    # ax.set_aspect('equal')
    if save_name != None:
        plt.savefig(save_name)
    
    plt.show()
    return fit_res
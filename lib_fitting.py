import numpy as np
import matplotlib.pyplot as plt

def Direct_mth(xarr, yarr, N_points=1000):
    """Solves for polynomial coefficients directly using a Vandermonde matrix.

    Args:
        xarr (array-like): The x-coordinates of the data points.
        yarr (array-like): The y-coordinates of the data points.
        N_points (int, optional): The number of evaluation points to generate for the resulting curve. Defaults to 1000.
    """

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
        """Initializes the Lagrange interpolation object.

        Args:
            x (array-like): The x-coordinates of the known data points.
            f (array-like): The y-coordinates (function values) of the known data points.
        """

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
        """Evaluates the Lagrange polynomial at a given coordinate.

        Args:
            x0 (float or array-like): The x-coordinate(s) at which to evaluate the interpolation.
        """

        frac = self.w / (x0 - self.x)
        return np.sum(self.f * frac) / np.sum(frac)
    

def lagrange_int(x, f, N_points=1000):
    """Generates coordinates for a smooth curve using Lagrange interpolation.

    Args:
        x (array-like): The x-coordinates of the reference data points.
        f (array-like): The y-coordinates of the reference data points.
        N_points (int, optional): The number of interpolated points to generate. Defaults to 1000.
    """

    lag = Lagrange(x, f)
    xoff = 0.001*(max(x) - min(x))
    x_in = np.linspace(x[0]-xoff, x[-1]+xoff, N_points)
    y_in = []
    for i in x_in:
        y_in.append(lag(i))
    return x_in, np.array(y_in)

def Cheby_nodes(dim):
    """Calculates Chebyshev nodes for a given dimension in the domain [-1, 1].

    Args:
        dim (int): The number of Chebyshev nodes to generate.
    """

    j = np.arange(dim)
    return -np.cos(j*np.pi / (dim-1))


def fit_engine(x, y, covar, poly_order):
    """Performs a generalized linear least-squares fit for a polynomial.

    Args:
        x (array-like): The independent variable data points.
        y (array-like): The dependent variable data points.
        covar (array-like): The covariance matrix of the y-data.
        poly_order (int): The degree of the polynomial to fit.
    """

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
    """Propagates parameters and their covariance matrix through a set of functions using the Jacobian.

    Args:
        par_func (list of callables): A list of functions defining the new parameters in terms of the old ones.
        old_pars (array-like): The original parameter values.
        old_cov_pars (array-like): The covariance matrix of the original parameters.
    """

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
    """Plots a fitted function against scatter data, complete with error bars and a confidence band.

    Args:
        f (callable): The fit function to plot.
        x_sc (array-like): The x-coordinates of the data points.
        y_sc (array-like): The y-coordinates of the data points.
        yerr (array-like): The standard errors of the y-coordinates.
        params (array-like): The fitted parameters to pass to the function f.
        cov_par (array-like): The covariance matrix of the fitted parameters.
        start (float): The starting x-value for the plotted fit line.
        stop (float): The ending x-value for the plotted fit line.
        fit_name (str, optional): The legend label for the fit line. Defaults to 'Fit function'.
        nsigma (float, optional): The number of standard deviations for the error band width. Defaults to 1.
        xlab (str, optional): The label for the x-axis. Defaults to 'X COO'.
        ylab (str, optional): The label for the y-axis. Defaults to 'Y COO'.
        save_name (str, optional): The file path to save the generated plot. Defaults to 'plots/plot.png'.
    """
    import lib_plot

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





def lin_fit (x, y, cov=None, poly_order=1, plus_minus=False, name_pars=None,
             mar='.', col='red', name_points='name points',
             labels=[r'$x$', r'$y$'], nsigma=3, lims=None, 
             ErrTrue=False, save_name=None, write_formula=False,
             figsz=(8, 5), true_val_student=None):
    """Executes a polynomial fit and plots the results with customizable styling and confidence intervals.

    Args:
        x (array-like): The independent variable data points.
        y (array-like): The dependent variable data points.
        cov (array-like, optional): The covariance matrix of the data. Default to identity matrix of len(x).
        poly_order (int, optional): The order of the polynomial to fit. Default to 1.
        plus_minus (bool, optional): If True the errors on the parameters are displayed on the plot, else not, useful for non statitistical relevant fit. Default to False.
        name_pars (list of str, optional): Parameter names to print in the console output. Defaults to None.
        mar (str, optional): The matplotlib marker style for the data points. Defaults to '.'.
        col (str, optional): The color used for the plot elements. Defaults to 'red'.
        name_points (str, optional): The legend label for the scatter data. Defaults to 'name points'.
        labels (list of str, optional): The axis labels [x_label, y_label]. Defaults to [r'$x$', r'$y$'].
        nsigma (float, optional): The number of standard deviations for the error band. Defaults to 3.
        lims (list or tuple, optional): The [min, max] limits for both x and y axes. Defaults to None.
        ErrTrue (bool, optional): If True, plots error bars and the confidence band. Defaults to True.
        save_name (str, optional): The file path to save the plot image. Defaults to None.
        write_formula (bool, optional): If True, includes the polynomial formula and values in the legend. Defaults to False.
        figsz (tuple, optional): Size of the figure plotted in the form of (x, y). Default to (8, 5)
        true_val_student (array-like): Array of the expected values for the parameters of the fit for a t-Student test, if None test is not performed. Default to None
    """
    import lib_plot
    import matplotlib.patches as mpatches
    from scipy import stats


    if len(x)==1:
        raise IndexError(f'Interpolation for {name_points} is not possible, there\'s only one point!!')

    if len(x) != len(y):
        raise IndexError(f'x and y do not have the same lenght! The method does not work.') 

    if cov == None:
        cov = np.eye(len(x))

    yerr = np.sqrt(np.diag(cov))
    fit_res = fit_engine(x, y, cov, poly_order)
    pars, cov_pars, chi2, pval = fit_res
    err_pars = np.sqrt(np.diag(cov_pars))
    
    # print on screen results of the fit
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if name_pars != None:
        for i, name in enumerate(name_pars):
            print(f'{name} = {pars[i]:.2e} +/- {err_pars[i]:.3e}')
        print()
    else:
        for i, par in enumerate(pars):
            print(f'{letters[i]} = {par:.2e} +/- {err_pars[i]:.3e}')
        print()

    print(f'Chi2 = {chi2:.2f}')
    print(f'p-val = {pval:.2f}')

    if true_val_student is not None:
        print('\n\n-----------------------------------------')
        print('t-Student TEST:')
        for i in range(len(pars)):
            true_val = true_val_student[i]
            exp_val = pars[i]
            sigma = err_pars[i]
            dof = len(x) - len(pars)
            
            t_val = abs(exp_val - true_val) / sigma
            
            p_value_t = 2 * stats.t.sf(t_val, dof)
            
            print(f'\nParameter {letters[i]}:')
            print(f'  Expected = {true_val}')
            print(f'  Fit      = {exp_val} ± {sigma}')
            print(f'  t-obs    = {t_val:.4f}')
            print(f'  p-value  = {p_value_t:.4f} --> {p_value_t*100:.2f}%')
        print('-----------------------------------------')


    # from a list of linear functions: [a, b*x, c*x^2, ...]
    # extract a single lambda func that represent: f(x) = a + b*x + c*x^2 + ...
    def f(x_coo, parameters):
        sol_ls = [np.flip(parameters)[i] * x_coo**i for i in range(len(parameters))]
        return np.sum(np.array(sol_ls), axis=0)

    eps = 1e-14
    start, stop = [min(x), max(x)]
    x_plot = np.linspace(start, stop, 500)
    y_plot = f(x_plot, pars)
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



    fig, ax = plt.subplots(figsize=figsz)
    # plot of:
    #     - Fit 
    #     - Points w/Errorbar
    #     - Error Band

    ax.plot(x_plot, f(x_plot, pars), label='Fit', color=col,
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
    lab_xy = [labels[0], labels[1]]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    handles, labels = plt.gca().get_legend_handles_labels()
    
    def add_empty_handle(label_text):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False))
        labels.append(label_text)

    # fit formula
    if write_formula:
            x_label_clean = lab_xy[0].replace('$', '')
            
            fit_form = rf'f({x_label_clean}) = '
            for i in np.arange(poly_order, 0, -1): 
                if i == poly_order:
                    if i == 1:    
                        fit_form += f' {letters[len(pars)-i-1]}{x_label_clean}' # Aggiunte graffe per esponenti > 9
                    else:
                        fit_form += f' {letters[len(pars)-i-1]}{x_label_clean}^{{{i}}}' # Aggiunte graffe per esponenti > 9
                elif i == 1:
                    fit_form += f' + {letters[len(pars)-2]}{x_label_clean}'
                else:
                    fit_form += f' + {letters[len(pars)-i-1]}{x_label_clean}^{{{i}}}' # Aggiunte graffe per esponenti > 9
            fit_form += rf'+' + letters[len(pars)-1]
            
            add_empty_handle(f'${fit_form}$')
    
    for i in range(len(pars)):
        if plus_minus:
            par_lab = rf'${letters[i]} = {pars[i]:.2f} \pm {err_pars[i]:.2f}$'
        else:
            par_lab = rf'${letters[i]} = {pars[i]:.2f}$'
        
        add_empty_handle(par_lab)
    
    
    ax.legend(handles=handles, labels=labels)
    
    plt.tight_layout()
    # ax.set_aspect('equal')
    if save_name != None:
        plt.savefig(save_name)
    
    plt.show()
    return fit_res






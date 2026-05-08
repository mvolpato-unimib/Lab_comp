import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('..'))

# ------------------------------------
#			  Graphics
# ------------------------------------
# If a better plot is needed, these settings allow for high quality image and text rendering

# plt.rcParams.update({
#     "text.usetex": True,           # Activate the use of LaTeX for all text
#     "font.family": "serif",        # Use a Serif font for normal text
#     "font.serif": ["Palatino"],    # Specify Palatino (very similar to the one in the photo)
#     "axes.labelsize": 16,          # Font size for axis labels
#     "font.size": 14,               # General font size
#     "legend.fontsize": 14,         # Font size for legend
#     "xtick.labelsize": 13,         # Font size for x-axis numbers
#     "ytick.labelsize": 13,         # Font size for y-axis numbers
#     "figure.figsize": (8, 6),      # Default figure size
#     "figure.dpi": 100,             # Resolution
#     "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
# })

# ------------------------------------




# ------------------------------------
# 			 Parameters
# ------------------------------------

N_points = 128
mL = 12             # represent the de Broglie lenght of the particle. The well has a width of mL dB lenghts.
BoundCond = 'open'  # boundary conditions
selec = 4           # select a certain number of first eigenvals to plot
n_mL = .5           # parameter to set the ylim. If ylim is not needed it is possible to comment the entire line:
                    # "plt.ylim(min(eig_val) - n_mL*mL, max(eig_val) + n_mL*mL)"
V0_L = 10

# ------------------------------------






# ------------------------------------
#			Static potential
# ------------------------------------
# GENERIC STATIC POTENTIAL:
# def StaticV(x):
#     return ...
# --------------------
# --------------------

# SOME EXAMPLES TO USE:
# def finite_well(x):
# 	return np.where((x < N_points/4) | (x > N_points * (3/4)), 0, -V0_L)

# def double_well(x):
# 	u = (x - N_points/2) / 8
# 	return V0_L * (u**4/2 - 2*u**2)

# def sin_well(x):
# 	x = np.asarray(x)
# 	return 2 * np.sin(x/max(x) * (2*np.pi))

# def triangular_well(x):
# 	v = np.where(x < N_points/4, V0_L*1e4, 0.5 * (x - N_points/4))
# 	return np.clip(v, -V0_L, V0_L*2)

# def alpha_well(x):
# 	R = N_points/4
	# coulomb = 100 / (x - R + 1e-10) 
	# return np.where(x < R, - V0_L, coulomb)

# ------------------------------------
# ------------------------------------

def StaticV(x):
	return np.where((x < N_points/4) | (x > N_points * (3/4)), 0, -V0_L)


def main():
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from lib_algebra import QR_eigensolver

    def hamiltonian(N, BC, V0):
        from lib_algebra import tridiagonal
        if BC not in ['open', 'close']:
            raise ValueError("Boundary conditions (BC) must be chosen between 'open' (A=B=0) and 'close' (A=B=1).")

        A = 0 if BC == 'open' else 1
        B = A

        K = - tridiagonal(N)
        K[0, -1], K[-1, 0] = (A, B)
        

        coor = np.arange(0, N)
        V = np.eye(N) * V0(coor)

        H = - N**2 / (2*mL) * K + V
        return H

    Hamilt = hamiltonian(N_points, BoundCond, StaticV)
    print('\nHamitonian produced, ready to find eigens...')
    # tot_eig_val, tot_eig_vect = QR_eigensolver(Hamilt, tol=1e-9, N_max=200)
    tot_eig_val, tot_eig_vect = np.linalg.eigh(Hamilt)



# ------------------------------------
#              Plotting
# ------------------------------------
    
    eig_val = np.copy(tot_eig_val)[:selec]
    eig_vect = (np.copy(tot_eig_vect)[:, :selec].T)**2
    print('\nEigenvalues =', eig_val)

    xcoo = np.arange(0, N_points)
    cols = [f'C{i}' for i in range(10)]

    plt.figure(figsize=(10, 6))
    plt.plot(xcoo, StaticV(xcoo), color='black', ls='--', alpha=0.7, 
            lw=2)
    for i, eig in enumerate(eig_val):
        plt.hlines(eig, min(xcoo), max(xcoo), color=cols[i], ls=':', alpha=0.6, lw=2)
        plt.plot(xcoo, (mL**2)*eig_vect[i] + eig, color=cols[i], zorder=2,
                label=rf'$\phi_{i}$')
        # symmetry
        # plt.plot(np.flip(xcoo), (mL**2)*eig_vect[i] + eig, color=cols[i], ls='--', zorder=2)

    # plt.vlines(xcoo[-1]/2, np.min(eig_vect)*mL+eig_val[0], np.max(eig_vect)*mL+eig_val[-1],
    #         color='black', ls=':', zorder=0, alpha=0.8)

    plt.ylim(min(eig_val) - n_mL*mL, max(eig_val) + n_mL*mL)

    plt.xlabel(r'$x/a$')
    plt.ylabel(r'$E$')
    plt.legend(loc='lower right')

    # plt.savefig('plots/schr_static_sol.png')
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np

# parameters
N_points = 64
V0_L = 10
mL = 8      # represent the de Broglie lenght of the particle. The well has a width of mL dB lenghts.
BoundCond = 'open'

vals, vects = ([], [])

def finite_well(x):
    return np.where((x < N_points/4) | (x > N_points * (3/4)), 0, -V0_L)

def double_well(x):
    u = (x - N_points/2) / 8
    return V0_L * (u**4/2 - 2*u**2)

def triangular_well(x):
    v = np.where(x < N_points/4, V0_L*1e4, 0.5 * (x - N_points/4))
    return np.clip(v, -V0_L, V0_L*2)

def alpha_well(x):
    R = N_points/4
    coulomb = 20 / (x - 15 + 1e-10) 
    return np.where(x < R, -V0_L, coulomb)

static_V_ls = [finite_well, double_well, triangular_well, alpha_well]
names = ['Finite well', 'Double well', 'Triangular well', 'Alpha well']

def main():

    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from lib_algebra import QR_eigensolver

    for i, StaticV in enumerate(static_V_ls):
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
        print(f'''\n-----------------------------------
        {names[i]}
-----------------------------------''')
        print('Hamitonian produced, ready to find eigens...')
        # tot_eig_val, tot_eig_vect = QR_eigensolver(Hamilt, tol=1e-9, N_max=200)
        tot_eig_val, tot_eig_vect = np.linalg.eigh(Hamilt)
        vals.append(tot_eig_val)
        vects.append(tot_eig_vect)
        print('Eigens found!')

    np.save('diff_val.npy', vals)
    np.save('diff_vect.npy', vects)
    print('\n\nResults written on txt files...\n')

if __name__ == "__main__":
    main()

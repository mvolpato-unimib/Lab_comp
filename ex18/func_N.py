import numpy as np

# parameters
V0_L = 10
mL = 8      # represent the de Broglie lenght of the particle. The well has a width of mL dB lenghts.
BoundCond = 'open'
N_ls = np.arange(16, 65, 4)
orders = [0, 3, 7] 

vals, vects = ([], [])
def main():

    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from lib_algebra import QR_eigensolver

    for k, N_points in enumerate(N_ls):
        def StaticV(x):
            x = np.asarray(x)
            return np.where((x < N_points/4) | (x >= 3/4 * N_points), 0, -V0_L)

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
        print(f'\nHamitonian N = {N_ls[k]} produced, ready to find eigens')

        # tot_eig_val, tot_eig_vect = QR_eigensolver(Hamilt, tol=1e-9, N_max=300)
        tot_eig_val, tot_eig_vect = np.linalg.eigh(Hamilt)

        vals.append([tot_eig_val[i] for i in orders])
    np.savetxt('scal_N_val.txt', vals)
    print('\n\nResults written on txt files...\n')

if __name__ == "__main__":
    main()

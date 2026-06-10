import numpy as np

# parameters
N_points = 64
V0_L = -10
mL = 8      # represent the de Broglie lenght of the particle. The well has a width of mL dB lenghts.
BoundCond = 'open'

def StaticV(x):
    """This function represent the potential in the Hamiltonian, expressed as dimensionless, so we do not
    use V, instead V --> VL. The result is that the subdivision of the interval for the V is independent from L
    which will never be expressed explicitly.\\
    The interval will always be expressed as:\\
    x =  0 --- 1/4 L --- --- 3/4 L --- L \\
    V =  -- 0 -------- -V0 -------- 0 --
    """

    x = np.asarray(x)
    return np.where((x < N_points/4) | (x >= 3/4 * N_points), 0, -V0_L)

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
    tot_eig_val, tot_eig_vect = QR_eigensolver(Hamilt, tol=1e-9, N_max=200)
    print('\nEigens found!')

    np.savetxt('mV_values.txt', tot_eig_val)
    np.savetxt('mV_vectors.txt', tot_eig_vect)
    print('Results written on txt files...\n')

if __name__ == "__main__":
    main()

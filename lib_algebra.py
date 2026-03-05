import numpy as np

def check_piv (A_in, indexes):
# Check if a given pivot row works out or gives problem with division
    L = np.copy(A_in)
    row, col = indexes
    if L[row][col] == 0:
        for i in range(1, np.shape(L)[0]):
            if L[i][col] != 0:
                piv = i
                r1, r2 = 0, piv
                L[[r1,r2]] =L[[r2,r1]]
                print(f'Changed initial matrix L by pivoting rows {row} and {piv} because:\nA[{row},{[piv]}] = 0\n')
                print(L, '\n')  
                break

def forw_subst_dumb(L, b, speak=False):
# FORWARD SUBSTITUTION ALGORITHM

    # Must be a square matrix
    if np.shape(L)[0] != np.shape(L)[1]:
        raise ValueError("Must use a square matrix!")

    n = np.shape(L)[0]
    np_xi = np.array[n]

    for i in range(n):
        # using dot prod we can optimize the func
        # we should see also entries useless but in np_xi vector they are 0,
        # so they do not interfere with the computation algorithm
        sec1 = max(i-1, 1)
        sec2 = max(i-2, 1)
        x_i = (b[i] - np.sum(np.dot(L[:sec1][:sec2], np_xi[:sec1]))) / L[i][i]
        np_xi[i] = x_i

    if speak:
        print('\nSolution found:\nx =', np_xi)
            
        print('\nTest true answer:')
        print('b =      ', b)
        print('L • x =  ', np.dot(L, np_xi))

def back_subst_dumb(L, b, speak=False):
# BACK SUBSTITUTION ALGORITHM

    # Must be a square matrix
    if np.shape(L)[0] != np.shape(L)[1]:
        raise ValueError("Must use a square matrix!")

    n = np.shape(L)[0] - 1
    x_n = b[n] / L[n, n]
    np_xi = np.zeros(n + 1)
    np_xi[n] = x_n

    for i in np.flip(np.arange(n)):
        # using dot prod we can optimize the func
        # we should see also entries useless but in np_xi vector they are 0,
        # so they do not interfere with the computation algorithm
        x_i = (b[i] - np.sum(np.dot(L[i], np_xi))) / L[i][i]
        np_xi[i] = x_i

    if speak:
        print('\nSolution found:\nx =', np_xi)
            
        print('\nTest true answer:')
        print('b =      ', b)
        print('L • x =  ', np.dot(L, np_xi))


def gauss_eli(A_in): 
# Function to perform GAUSSIAN ELIMINATION

    # the function do not touch the original matrix
    L = np.copy(A_in)
    n = np.shape(L)[0]
    for j in range(n-1):
        # print('Col =', j)
        for i in range(j+1, n):
            c = L[i,j] / L[j,j]
            L[i,:] -= c * L[j,:]

        #     print(f'Result on row {i}:\n{L}')
        # print()
    print(f'Reducted matrix by Gauss. red.:\n{L}')  
    return L


def chol_fact(L):
# Returns the Cholesky Factor L

    n = np.shape(L)
    # Must be a square matrix
    if L.shape[0] != L.shape[1]:
        raise ValueError("Must use a square matrix!")

    # Check if symmetric
    if not np.allclose(L, L.T, atol=1e-8):
        raise ValueError(f'L is not symmetric! Max distance between matrices: {np.max(np.abs(L - L.T)):.2e}')

    L = np.zeros(shape=n, dtype=np.float64)
    for i in range(n[0]):
        for j in range(n[0]):
            if i == j:
            # Diagonal elements
                L[i][i] = np.sqrt(L[i][i] - np.sum(L[i][:i]**2))
            elif i > j:
            # Off diagonal
                sec = max(i-2, 1)
                L[i][j] = 1 / L[j][j] * (L[i][j] - np.sum(np.dot(L[i][:sec], L[j][:sec])))
    print('\nCholesky factor (L):\n', L)
    return L

def BackChol_dumb(L, b):
    L = chol_fact(L)
    back_subst_dumb(L, b)


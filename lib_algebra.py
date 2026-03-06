import numpy as np

def forw_subst(L_in, b_in, speak=False):
    L = np.copy(L_in)
    b = np.copy(b_in)
    if np.shape(L)[0] != np.shape(L)[1]:
        raise ValueError("Must use a square matrix!")

    n = np.shape(L)[0]
    np_xi = np.zeros(n)

    for i in range(n):
        x_i = (b[i] - np.dot(L[i, :i], np_xi[:i])) / L[i, i]
        np_xi[i] = x_i

    test = np.allclose(np.dot(L, np_xi), b) 
    if speak:
        print('\nSolution found:\nx =', np_xi)
        print(f'\nTest true answer: {test}')
        # print('b =      ', b)
        # print('L • x =  ', np.dot(L, np_xi))
    return np_xi

def back_subst(U_in, b_in, speak=False):
    U = np.copy(U_in)
    b = np.copy(b_in)
    if np.shape(U)[0] != np.shape(U)[1]:
        raise ValueError("Must use a square matrix!")

    n = np.shape(U)[0]
    np_xi = np.zeros(n)
    
    for i in reversed(range(n)):
        x_i = (b[i] - np.dot(U[i, i+1:], np_xi[i+1:])) / U[i, i]
        np_xi[i] = x_i

    test = np.allclose(np.dot(U, np_xi), b) 
    if speak:
        print('\nSolution found:\nx =', np_xi)
        print(f'\nTest true answer: {test}')    # Prints True if the answer is correct after the check with b
        # print('b =      ', b)
        # print('U • x =  ', np.dot(U, np_xi))
                
    return np_xi

def BackGauss_dumb(A_in, b_in, speak=False): 
    A = np.copy(A_in)
    b = np.copy(b_in)
    n = np.shape(A)[0]
    for j in range(n-1):
        for i in range(j+1, n):
            c = A[i,j] / A[j,j]
            A[i,:] -= c * A[j,:]
            b[i] -= c * b[j]  
    return back_subst(A, b, speak)

def matrix_inverse_dumb(A, speak=False):
# Computes inverse of a matrix
    n = A.shape[0]
    A_inv = np.zeros_like(A, dtype=float)
    
    # Cycle on each of Id mat cols
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1  # i-col of Id mat
        # solve A x = e using gaussian elimination + backward
        x = BackGauss_dumb(A.copy(), e)
        # solution becomes col of inverse
        A_inv[:, i] = x
    
    if speak:
        test = np.allclose(A @ A_inv, np.eye(A.shape[0]))
        print('Solution found:\nA^-1 =\n', A_inv)
        print(f'\nTest true answer: {test}')    # Prints True if the answer is correct
    return A_inv

def BackGauss(A_in, b_in, speak=False):
    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    n = np.shape(A)[0]

    for j in range(n-1):
        # Find row with maximum absolute value in column j
        max_row = j + np.argmax(np.abs(A[j:, j]))
        if max_row != j:
            A[[j, max_row]] = A[[max_row, j]]
            b[[j, max_row]] = b[[max_row, j]]
            # print(f"Pivoting: swapped row {j} with row {max_row}")
        
        # Gaussian elimination
        for i in range(j+1, n):
            c = A[i,j] / A[j,j]
            A[i,:] -= c * A[j,:]
            b[i] -= c * b[j]
            
    # Backward substitution
    return back_subst(A, b, speak)

def matrix_inverse(A, speak=False):
# Computes inverse of a matrix
    n = A.shape[0]
    A_inv = np.zeros_like(A, dtype=float)
    
    # Cycle on each of Id mat cols
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1  # i-col of Id mat
        # solve A x = e using gaussian elimination + backward
        x = BackGauss(A.copy(), e)
        # solution becomes col of inverse
        A_inv[:, i] = x
    
    if speak:
        test = np.allclose(A @ A_inv, np.eye(A.shape[0]))
        print('Solution found:\nA^-1 =\n', A_inv)
        print(f'\nTest true answer: {test}')    # Prints True if the answer is correct
    return A_inv



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

def BackChol(L, b):
    L = chol_fact(L)
    back_subst(L, b)


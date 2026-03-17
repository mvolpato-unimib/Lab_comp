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
    
    if not test:
        raise ValueError("Algorithm fails!\nb != L • x")
    
    if speak:
        print('\nSolution found:\nx =', np_xi)
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

    if not test:
        raise ValueError("Algorithm fails!\nb != U • x")

    if speak:
        print('\nSolution found:\nx =', np_xi)
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
    
    Mat_id = np.eye(n)
    # Cycle on each of Id mat cols
    for i in range(n):
        e = Mat_id[i]  # i-col of Id mat
        # solve A x = e using gaussian elimination + backward
        x = BackGauss(A.copy(), e)
        # solution becomes col of inverse
        A_inv[:, i] = x
    
    test = np.allclose(A @ A_inv, np.eye(A.shape[0]))
    if not test:
        raise ValueError("Algorithm fails!\nA - A^-1 != 0")

    if speak:
        print('Solution found:\nA^-1 =\n', A_inv)
    return A_inv


def chol_fact(A):
    n = np.shape(A)
    # Must be a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Must use a square matrix!")

    # Check if symmetric
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError(f'A is not symmetric! Max distance between matrices: {np.max(np.abs(A - A.T)):.2e}')

    L = np.zeros(shape=n, dtype=np.float64)
    for i in range(n[0]):
        for j in range(n[0]):
            if i == j:
            # Diagonal elements
                L[i][i] = np.sqrt(A[i][i] - np.sum(L[i][:i]**2))
            elif i > j:
            # Off diagonal
                L[i, j] = (A[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
    test = np.allclose(L @ L.conj().T, A)
    if not test:
        raise ValueError("Algorithm fails!Il test L*L^T != A")
    
    return L

def BackChol(A, b):
    L = chol_fact(A)
    y = forw_subst(L, b)
    x = back_subst(L.conj().T, y)
    return x


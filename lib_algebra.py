import numpy as np

# ----------------------------------------------------------
# LINEAR SYSTEMS SOLVER AND DECOMPOSITION OF MATRICES 
# ----------------------------------------------------------

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


def LU_dec(A_in):
    """Performs LU decomposition on a given A matrix.
    Registers also piv_sign, the sign of the Determinant given by the pivoting operations"""

    A = np.copy(A_in)
    n = np.shape(A)[0]
    L = np.eye(n)
    piv_sign = 1

    for j in range(n-1):
        # Find row with maximum absolute value in column j
        max_row = j + np.argmax(np.abs(A[j:, j]))
        if max_row != j:
            A[[j, max_row]] = A[[max_row, j]]
            if j > 0:
                L[[j, max_row], :j] = L[[max_row, j], :j]
            piv_sign *= -1


        # Gaussian elimination
        for i in range(j+1, n):
            c = A[i,j] / A[j,j]
            A[i,j:] -= c * A[j,j:]
            L[i, j] = c
    
    return L, A, piv_sign # A == U


def chol_fact(A):
    n = np.shape(A)
    # Must be a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Must use a square matrix!")

    # Check if symmetric
    if not np.allclose(A, A.T.conj(), atol=1e-8):
        raise ValueError(f'A is not symmetric! Max distance between matrices: {np.max(np.abs(A - A.T.conj())):.2e}')

    L = np.zeros(shape=n, dtype=np.float64)
    for i in range(n[0]):
        for j in range(n[0]):
            if i == j:
            # Diagonal elements
                L[i][i] = np.sqrt(A[i][i] - np.sum(L[i][:i]**2))
            elif i > j:
            # Off diagonal
                L[i, j] = (A[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
    test = np.allclose(L @ L.T.conj(), A)
    if not test:
        raise ValueError("Algorithm fails!Il test L*L^T != A")
    
    return L


def BackChol(A, b):
    L = chol_fact(A)
    y = forw_subst(L, b)
    x = back_subst(L.T.conj(), y)
    return x


def householder_2x2(A_input):
    """returns the Projector P_v for a matrix 2x2, householder projector method"""
    A = np.copy(A_input)
    u = np.copy(A[:,0])  # initially set as first column of A and then added the term plus/minus
    n = np.sqrt(np.real(u.T.conj() @ u))
    s = 1
    if A[0,0] - n > A[0,0] + n:
        s = - 1
    u[0] += s * n

    den = n**2 + s * A[0,0] * n
    P = np.eye(len(A)) - np.outer(u,u) / den
    return P

def QR_dec(A_input):
    dim = len(A_input)
    R = np.copy(A_input)
    Q = np.eye(dim)

    for i in range(dim-1):
        x = R[i:, i]
        e1 = np.zeros_like(x)
        e1[0] = np.sqrt(np.real(x.T.conj() @ x)) * (1 if x[0] >= 0 else -1)

        u = x + e1
        v = u / np.sqrt(np.real(u.T.conj() @ u))

        Pi = np.eye(dim - i) - 2 * np.outer(v, v)

        P = np.eye(dim)
        P[i:, i:] = Pi

        R = P @ R
        Q = Q @ P.T.conj()

    if not np.allclose(A_input, Q@R):
        raise ValueError('The decomposition is not working! A - A\' != 0')
    return Q, R


def QR_solver(A, b):
    """
    Solve a linear system using QR decomposition and Backward substitution.
    """
    Q,R = QR_dec(A)
    x_sol = back_subst(R, Q.T.conj() @ b)
    return x_sol 

# ----------------------------------------------------------
# END SOLVER
# ----------------------------------------------------------



# ----------------------------------------------------------
# DETERMINANT ALGORITHMS
# ----------------------------------------------------------
def determinant(A_in, choice='QR'):
    """"
    Evaluate the determinant of a given matrix choosing between different methods:
    - LU decomposition
    - Cholesky decomposition
    - QR decomposition

    The standard is QR, chosen for the best stability
    """

    if choice=='LU':
        """
        Evaluate the determinant of a given generic matrix A
        """
        n = len(A_in)
        L, U, s_piv = LU_dec(A_in)
        det = s_piv
        for row in range(n):
            det *= U[row,row]    

        return det


    if choice=='Cholesky':
        """
        Evaluate the determinant of a given symmetric matrix A
        """
        # Check if symmetric
        if not np.allclose(A_in, A_in.T.conj(), atol=1e-8):
            raise ValueError(f'A is not symmetric! Max distance between matrices: {np.max(np.abs(A_in - A_in.T.conj())):.2e}')
        
        n = len(A_in)
        L = chol_fact(A_in)
        det = 1
        for row in range(n):
            det *= L[row,row]**2

        return det


    if choice=='QR':
        """
        Evaluate the determinant of a given generic matrix A
        """
        n = len(A_in)
        Q, R = QR_dec(A_in)
        det = (-1)**(n-1)
        for row in range(n):
            det *= R[row,row]     

        return det

# ----------------------------------------------------------
# END DETERMINANT
# ----------------------------------------------------------



# ----------------------------------------------------------
# INVERSE OF A MATRIX
# ----------------------------------------------------------

def matrix_inverse(A, speak=False):
    """
    Computes inverse of a matrix.
    """
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

# ----------------------------------------------------------
# END INVERSE
# ----------------------------------------------------------





# ----------------------------------------------------------
# EIGENVALUES & EIGENVECTORS
# ----------------------------------------------------------

def Power_mth(A_in, epsilon=1e-14, N_max=100000):
    """
    Finds maximum eigval and eigvect for a given Real matrix.
    Doesn't work for Complex matrix.
    """
    A = np.copy(A_in)
    x_0 = np.random.rand(len(A))    # choice of a random vector
    y = x_0 / np.sqrt(np.real(x_0.T.conj() @ x_0))      # the vector is chosen normalised
    lamb1 = 1
    lambt = 0

    for i in range(N_max):
        if abs(lamb1 - lambt) > epsilon:
            y = A @ y
            y = y / np.sqrt(y@y)
            lambt = lamb1
            lamb1 = y.T.conj() @ A @ y     # y is normalised --> lamb1 is the next eigval
        else:
            # print(f'Iteration {i}: reached max precision\n')
            break
        
    eigVal = lamb1 
    eigVect = y
    if np.sum(A @ eigVect) - np.sum(eigVal * eigVect) > 1e-8:
        raise ValueError('Eigenvalue/vector is not accurate')

    return eigVal, eigVect



# ----------------------------------------------------------
# END EIGENS
# ----------------------------------------------------------




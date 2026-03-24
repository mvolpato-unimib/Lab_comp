#include <stdio.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <tuple>
#include "matrix.h"

template<typename T> struct is_complex : std::false_type {};
template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
using Vector = std::vector<T>;
using dMatrix = Matrix<double>;
using cMatrix = Matrix<std::complex<double>>;
using std::cout;


// ----------------------------------------------------------
// LINEAR SYSTEMS SOLVERS AND DECOMPOSITION OF MATRICES 
// ----------------------------------------------------------

/**
 * The function performs Forward Substitution on a matrix L
 * @param L Lower triangular matrix
 * @param b Vector of known terms
 * @return Matrix<T>(n,1) Vector of solutions
 */
template <typename T>
Matrix<T> ForwSubst(const Matrix<T>& L, 
                    const Matrix<T>& b) {
    int n = L.nc;
    Matrix<T> x(n, 1);

    for (int i = 0; i < n; ++i) {
        T sum = 0;
        for (int j = 0; j < i; ++j) {
            sum += L(i,j) * x(j,0);
        }
        x(i,0) = (b(i,0) - sum) / L(i,i);
    }
    return x;
}


/**
 * The function performs Backward Substitution on a matrix U
 * @param U Upper triangular matrix
 * @param b Vector of known terms
 * @return Matrix<T>(n,1) Vector of solutions
 */
template <typename T>
Matrix<T> BackSubst(const Matrix<T>& U, 
                    const Matrix<T>& b) {
    int n = U.nc;
    Matrix<T> x(n, 1);

    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += U(i, j) * x(j, 0);
        }
        x(i, 0) = (b(i, 0) - sum) / U(i, i);
    }
    return x;
}


/**
 * Performs LU decomposition on a given A matrix.
 * Registers also piv_sign, the sign of the Determinant given by the pivoting operations
 * @param A Generic square matrix
 * @return L Lower matrix, U Upper matrix, piv_sign Sign of the determinant
 */
template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<int>, int> 
lu_decomposition(const Matrix<T>& A) {
    int n = A.nr;
    Matrix<T> L(n,n);
    L.identity();
    Matrix<T> U(A);
    Matrix<int> P(n,1);
    for (int i=0; i<n; i++) {
        P(i,0) = i;
    }
    int piv_sign = 1;

    // Find row with maximum absolute value in column j
    for (int j=0; j<n; j++) {
        int max_row = j;
        for (int i = j + 1; i < n; i++) {
            if (std::abs(U(i, j)) > std::abs(U(max_row, j))) {
                max_row = i;
            }
        }
        
        if (max_row != j) {
            U.swap_rows(j, max_row);
            P.swap_rows(j, max_row);
            piv_sign *= -1;
            
            for (int k = 0; k < j; k++) {
                std::swap(L(j, k), L(max_row, k));
            }
        }

        // Gaussian elimination
        for (int i=j+1; i<n; i++) {
            T c = U(i,j) / U(j,j);
            for (int k=j; k<n; k++) {
                U(i,k) -= c * U(j, k);
            }
            L(i,j) = c;
        }
    }
    
    return {L, U, P, piv_sign};
}


/**
 * Performs QR decomposition to a input matrix A:
 * @param A Generic input square matrix
 * @return Q, R square matrices:
 * - Q Orthogonal matrix
 * - R Upper triangular matrix
 */
template <typename T>
std::tuple<Matrix<T>, Matrix<T>> 
QR_dec(const Matrix<T>& A) {
    int n = A.nr;
    Matrix<T> R(A);
    Matrix<T> Q(n,n);
    Q.identity();

    for (int i=0; i<n-1; i++) {
        Matrix<T> x(n-i, 1);
        for (int k=i; k<n; k++) {
            x(k-i, 0) = R(k, i);
        }
        
        Matrix<T> e1(x); e1.zeros();
        double norm = std::sqrt(x.norm2());
        
        if constexpr (is_complex<T>::value) {
            std::complex<double> phase = x(0,0) / std::abs(x(0,0));
            e1(0,0) = norm*phase;
        } else {
            double sign = (x(0,0) >= 0) ? 1.0 : -1.0;
            e1(0,0) = sign * norm;
            e1(0,0) = norm;
        }

        Matrix<T> u = x + e1;
        Matrix<T> v = u * (1 / std::sqrt(u.norm2()));

        Matrix<T> idenPi(n-i, n-i);
        idenPi.identity();
        Matrix<T> outer(v.nr, v.nr);
        for (int row=0; row<v.nr; row++) {
            for (int col=0; col<v.nr; col++) {
                auto conj_val = v(col, 0);
                if constexpr (is_complex<T>::value) {
                    outer(row, col) = v(row, 0) * std::conj(conj_val);
                } else {
                    outer(row, col) = v(row, 0) * conj_val;
                }
            }
        }
        
        Matrix<T> Pi = idenPi - outer * 2;
        Matrix<T> P(n,n);
        P.identity();
        for (int row=i; row<n; row++) {
            for (int col=i; col<n; col++) {
                P(row,col) = Pi(row - i, col - i);
            }
        }

        R = P.dot(R);
        Q = Q.dot(P.dagger());
    }

    return {Q, R};
}



/**
 * Solve the generic linear system using QR decomposition combined with Backward substitution
 * @param A Generic square matrix of the coefficients
 * @param b Vector of known terms
 * @result x Vector of the solutions 
 */
template <typename T>
Matrix<T> QR_solver(const Matrix<T>& A, const Matrix<T>& b) {
    auto [Q, R] = QR_dec(A);
    auto x = BackSubst(R, Q.dagger().dot(b));  
    return x;
}

// ----------------------------------------------------------
// END SOLVER
// ----------------------------------------------------------







// ----------------------------------------------------------
// DETERMINANT ALGORITHMS
// ----------------------------------------------------------

// ----------------------------------------------------------
// END DETERMINANT
// ----------------------------------------------------------






// ----------------------------------------------------------
// INVERSE OF A MATRIX
// ----------------------------------------------------------


// ----------------------------------------------------------
// END INVERSE
// ----------------------------------------------------------







// # ----------------------------------------------------------
// # EIGENVALUES & EIGENVECTORS
// # ----------------------------------------------------------

// # ----------------------------------------------------------
// # END EIGENS
// # ----------------------------------------------------------

int main(){
    dMatrix A(3,3);
    // A.random();
    A.data = {2., -1, .0, 
            -1., 2.0, -1.0, 
            .0, -1.0, 2.0};

    cout << "Matrice A originale:\n";
    A.print();

    dMatrix b(3,1);
    b.data = {1, 0, 1};

    cout << "Matrice b originale:\n";
    b.print();

    auto x = QR_solver(A, b);
    cout << "\nMatrice x:\n"; x.print();

    using cpx = std::complex<double>;
    const cpx i(0, 1);

    // Matrice 2x2 complessa
    dMatrix A(2,2);
    A.data = {  i, 1.0, 
            2.0,  -i };



    // cout << "\nVerifica QR (Q*R - A):\n";
    // dMatrix diff = Q.dot(R) - A;
    // diff.print();

    return 0;
}


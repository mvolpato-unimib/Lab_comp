#include <stdio.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <tuple>
#include "matrix.h"

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


// QR DEC



// SOLVER

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
    A.random();
    cout << "Matrice A originale:\n";
    A.print();

    auto [L, U, P, det_sign] = lu_decomposition(A);

    cout << "\nMatrice L:\n"; L.print();
    cout << "\nMatrice U:\n"; U.print();

    // Verifichiamo PA = LU
    cout << "\nVerifica LU (Prodotto L*U):\n";
    dMatrix LU = L.dot(U);
    LU.print();

    cout << "\nVerifica PA (A riordinata secondo P):\n";
    // Creiamo una matrice che rappresenta A con le righe scambiate
    dMatrix PA(3,3);
    for(int i=0; i<3; i++) {
        int row_index = P(i, 0); // Prendi l'indice della riga originale
        for(int j=0; j<3; j++) {
            PA(i, j) = A(row_index, j);
        }
    }
    PA.print();

    return 0;
}


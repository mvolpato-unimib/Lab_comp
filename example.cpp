#include <iostream>
#include <complex>
#include "matrix.h"

// 1. Dichiarazione (o sposta sopra il main)
double inner_product(const Matrix<double> &metric, const Matrix<double> &vec);

int main() {
    // ... (resto del codice)

    Matrix<double> vec2(3,1);
    Matrix<double> mat(3,3);
    mat.identity();
    
    // 2. Adesso funziona
    double inner2 = inner_product(mat, vec2);
    
    return 0;
}

double inner_product(const Matrix<double> &metric, const Matrix<double> &vec) {
    // dot() restituisce una Matrix 1x1, prendiamo l'elemento (0,0)
    Matrix<double> res = vec.dagger().dot(metric.dot(vec));
    return res(0, 0); 
}
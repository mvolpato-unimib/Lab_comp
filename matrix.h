#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <vector>
#include <cstring>
#include <stdlib.h>
#include <complex>
#include <algorithm>


/*
Utility class to implement basic linear algebra operations among matrices and vectors.
*/

template<typename T> class Matrix {
   public:
      std::vector<T> data;
      int nr, nc;

      Matrix(int _nr, int _nc) : nr(_nr), nc(_nc), data(_nr*_nc,0) {};
      Matrix(const Matrix<T> &in) : nr(in.nr), nc(in.nc), data(in.data) {}
    
      int site(int i,int j) const {
         return i*nc + j;
      }

      T operator()(int i, int j) const {
         return data[this->site(i,j)];
      }

      T& operator()(int i,int j) {
         return data[this->site(i,j)];
      }

      void swap_rows(int i, int j) {
         T *tmp = new T[nc];
         memcpy(tmp, &data[this->site(i,0)], sizeof(T)*nc);
         memcpy(&data[this->site(i,0)], &data[this->site(j,0)], sizeof(T)*nc);
         memcpy(&data[this->site(j,0)], tmp, sizeof(T)*nc);
         delete[] tmp;
      }

      void ones() {
         std::fill(data.begin(), data.end(), 1.0);
      }

      void zeros() {
         std::fill(data.begin(), data.end(), 0);
      }
    
      void identity() {
         this->zeros();
         assert(nr==nc);
         for (int i=0;i<nc;i++)
            data[this->site(i,i)] = 1.0;
      }
    
      void random() {
         for (int i=0;i<nr*nc;i++)
            data[i] = (T)(rand() / (double)RAND_MAX);
      }

      void flip() {
         std::reverse(data.begin(), data.end());
      }

      Matrix<T> flipped() const {
         Matrix<T> out(*this);
         out.flip();
         return out;
      }

      Matrix<T>& operator=(const Matrix<T> &in) {
         nr = in.nr;
         nc = in.nc;
         data = in.data;
         return *this;
      }

      Matrix<T> operator+(const Matrix<T> &b) const {
         Matrix<T> out(nr, nc);
         assert((nr == b.nr)&&(nc == b.nc));
         for (int i=0;i<nr*nc;i++)
            out.data[i] = data[i] + b.data[i];
         return out;
      }

      Matrix<T> dot(const Matrix<T> &a) const {
         Matrix<T> out(nr,a.nc);
         assert((nc == a.nr));
         for (int i=0;i<nr;i++)
            for (int j=0;j<a.nc;j++)
               for (int k=0;k<nc;k++)
                  out(i,j) += data[this->site(i,k)] * a(k,j); 
         return out;
      }

      template<typename T2> Matrix<T> operator*(T2 d) const {
         Matrix<T> out(*this);
         for (int i=0;i<nr*nc;i++)
            out.data[i] *= d;
         return out;
      }

      Matrix<T> operator-(const Matrix<T> &b) const {
         return *this + (b * (-1.0));
      }

      Matrix<T> dagger() const {
         Matrix<T> out(nc, nr);
         for (int i = 0; i < nc; i++) {
            for (int j = 0; j < nr; j++) {
                  if constexpr (std::is_floating_point_v<T>) {
                     out(i, j) = data[this->site(j, i)]; 
                  } else {
                     using std::conj;
                     out(i, j) = conj(data[this->site(j, i)]);
                  }
            }
         }
         return out;
      }

      double norm2() {
         double n = 0.0;
         for (int i=0;i<nr*nc;i++) {
            double h = abs(this->data[i]);
            n += h*h;
         }
         return n;
      }

      void print() {
         for (int i=0;i<nr;i++) {
            for (int j=0;j<nc;j++)
               printf("% 02e ",data[this->site(i,j)]);
            printf("\n");
         }

      }
};


#endif

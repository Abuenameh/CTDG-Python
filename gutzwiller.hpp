/* 
 * File:   gutzwiller.hpp
 * Author: Abuenameh
 *
 * Created on August 10, 2014, 10:45 PM
 */

#ifndef GUTZWILLER_HPP
#define	GUTZWILLER_HPP

#include <complex>
#include <vector>
#include <iostream>

using namespace std;

const int L = 5;
const int nmax = 7;
const int dim = nmax + 1;

const double M = 1000;
const double g13 = 2.5e9;
const double g24 = 2.5e9;
const double delta = 1.0e12;
const double Delta = -2.0e10;
const double alpha = 1.1e7;

const double Ng = sqrt(M) * g13;

#include "casadi.hpp"

template<class T>
complex<T> operator~(const complex<T> a) {
	return conj(a);
}

struct Parameters {
    double theta;
};

inline int mod(int i) {
	return (i + L) % L;
}

inline SX g(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline double g2(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline SX ga(int n, int a) {
    return sqrt(1.0*(n+1)*(n-a+1));
}

//inline SX eps(vector<SX>& U, int i, int j, int n, int m) {
inline SX eps(SX& U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

inline SX eps(SX& U, int n, int m) {
    return (n - m + 1) * U;
}

//inline SX eps(vector<SX>& U, int i, int j, int n, int m, int k, int l, int p, int q) {
inline SX eps(SX& U, int i, int j, int n, int m, int k, int l, int p, int q) {
    return n*U[i] - (m-1)*U[j] + (q-1)*U[k] - p*U[l];
}

inline double eps(double U, int n, int m) {
    return (n - m + 1) * U;
}

inline SX eps(SX& U, int i, int j, int a) {
    return a*U;
}


#endif	/* GUTZWILLER_HPP */


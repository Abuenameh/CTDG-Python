#include "casadimath.hpp"
#include "gutzwiller.hpp"

namespace casadi {

    inline bool isnan(SX& sx) {
        return sx.at(0).isNan();
    }

    inline bool isinf(SX sx) {
        return sx.at(0).isInf();
    }
}

complex<SX> operator*(double x, complex<SX> sx) {
    return complex<SX>(x,0) * sx;
}

complex<SX> operator*(complex<SX> sx, double x) {
    return complex<SX>(x,0) * sx;
}

complex<SX> operator*(complex<SX> csx, SX sx) {
    return csx * complex<SX>(sx,0);
}

#include "casadimath.incl"

SX energyc(SX& fin, SX& J, SX& U0, SX& dU, double mu, bool normalize) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2*theta);
    double sin2th = sin(2*theta);
    
    complex<SX> E = complex<SX>(0,0);
    
    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;
    
#include "casadi.incl"
    
    return E.real();
}

SX energyc(SX& fin, SX& J, SX& U0, SX& dU, SX& mu, bool normalize) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2*theta);
    double sin2th = sin(2*theta);
    
    complex<SX> E = complex<SX>(0,0);
    
    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;
    
#include "casadi.incl"
    
    return E.real();
}

SX energyri(SX& fin, SX& J, SX& U0, SX& dU, SX& mu, bool normalize) {
//    vector<vector<SX>> fr(L, vector<SX>(dim, 0));
//    vector<vector<SX>> fi(L, vector<SX>(dim, 0));
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
//            fr[i][n] = fin[2*(i*dim+n)];
//            fi[i][n] = fin[2*(i*dim+n)+1];
//        }
//    }
//    
//    vector<SX> norm2(L, 0);
//    for (int j = 0; j < L; j++) {
//        for (int m = 0; m <= nmax; m++) {
//            norm2[j] += fr[j][m] * fr[j][m] + fi[j][m] * fi[j][m];
//        }
//    }
    
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2*theta);
    double sin2th = sin(2*theta);
    
//    SX E = 0;
    complex<SX> E = complex<SX>(0,0);
    
//    SX Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;
    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;
    
#include "casadi.incl"
    
    return E.real();
}

double energyri(vector<double> fin, vector<double> J, double U0, vector<double> dU, double mu) {
//    vector<vector<double>> fr(L, vector<double>(dim, 0));
//    vector<vector<double>> fi(L, vector<double>(dim, 0));
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
//            fr[i][n] = fin[2*(i*dim+n)];
//            fi[i][n] = fin[2*(i*dim+n)+1];
//        }
//    }
//    
//    double theta = 0;
//    double costh = cos(theta);
//    double sinth = sin(theta);
//    double cos2th = cos(2*theta);
//    double sin2th = sin(2*theta);
//    
//    double E = 0;
//    
//    double Ei;
//    double Ej1;
//    double Ej2;
//    double Ej1j2;
//    double Ej1k1;
//    double Ej2k2;
//    
//#include "casadi.incl"
//    
//    return E;
    return 0;
}
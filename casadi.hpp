/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <typeinfo>

#include <casadi/casadi.hpp>
#include <casadi/solvers/rk_integrator.hpp>
#include <casadi/solvers/collocation_integrator.hpp>
#include <casadi/interfaces/sundials/cvodes_interface.hpp>
#include <casadi/core/function/custom_function.hpp>

using namespace casadi;

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"

inline double JW(double W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

inline double JWij(double Wi, double Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

inline double UW(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

inline SX JW(SX W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

inline SX JWij(SX Wi, SX Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

inline SX UW(SX W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

//SX energy(SX& fin, SX& J, SX& U0, SX& dU, double mu);
//SX energy(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, double mu);
//SX canonical(SX& fin, SX& J, SX& U0, SX& dU, double mu);
//SX canonical(int i, int n, SX& f, SX& J, SX& U0, SX& dU, double mu);

template<class T> SX energy(SX& fin, SX& J, SX& U0, SX& dU, T mu);
template<class T> SX energy(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, T mu);
template<class T> SX canonical(SX& fin, SX& J, SX& U0, SX& dU, T mu);
template<class T> SX canonical(int i, int n, SX& f, SX& J, SX& U0, SX& dU, T mu);

template<class T> SX energy2(SX& fin, SX& J, SX& U0, SX& dU, T mu);
template<class T> SX energy2(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, T mu);

//SX energy(SX& fin, SX& J, SX& U0, SX& dU, SX& mu);
//SX energy(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, SX& mu);
//SX canonical(SX& fin, SX& J, SX& U0, SX& dU, SX& mu);
//SX canonical(int i, int n, SX& f, SX& J, SX& U0, SX& dU, SX& mu);

complex<double> dot(vector<complex<double>>&v, vector<complex<double>>&w);

//struct results {
//    double tau;
//    double Ei;
//    double Ef;
//    double Q;
//    double p;
//    //    vector<vector<double>> bs;
//    double U0;
//    vector<double> J0;
//    vector<complex<double>> b0;
//    vector<complex<double>> bf;
//    vector<vector<complex<double>>> f0;
//    vector<vector<complex<double>>> ff;
//    ptime begin;
//    ptime end;
//    string runtime;
//};

class DynamicsProblem {
public:
    DynamicsProblem(double Wi, double Wf, double mu, vector<double> xi);

    static void setup(double W, double mu, vector<double>& xi);

//    void evolve(double tau, results& res);

private:

    static SX energy(SX& fin, SX& J, SX& U0, SX& dU, double mu);
    static SX energy(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, double mu);
    static SX canonical(SX& fin, SX& J, SX& U0, SX& dU, double mu);
    static SX canonical(int i, int n, SX& f, SX& J, SX& U0, SX& dU, double mu);

    static double U00;
    static vector<double> J0;

    SXFunction E0;
    
    SXFunction ode_func;
    Integrator integrator;

    static vector<double> x0;

    static vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;

};

#endif	/* CASADI_HPP */


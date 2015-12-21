/* 
 * File:   casadiri.hpp
 * Author: Abuenameh
 *
 * Created on November 29, 2015, 11:11 PM
 */

#ifndef CASADIMATH_HPP
#define	CASADIMATH_HPP

#include "gutzwiller.hpp"

inline double eps(vector<double>& U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

SX energyri(SX& fin, SX& J, SX& U0, SX& dU, SX& mu, bool normalize);
double energyri(vector<double> fin, vector<double> J, double U0, vector<double> dU, double mu);
SX energyc(SX& fin, SX& J, SX& U0, SX& dU, double mu, bool normalize);
SX energyc(SX& fin, SX& J, SX& U0, SX& dU, SX& mu, bool normalize);

#include "casadimath.hincl"

#endif	/* CASADIRI_HPP */


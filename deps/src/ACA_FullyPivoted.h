#ifndef _ACA_FULLYPIVOTED_
#define _ACA_FULLYPIVOTED_

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <list>
#include <cassert>
#include <sys/time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>



using namespace std;
using namespace Eigen;

// class vec3;
struct oct_node;

void ACA_FullyPivoted(MatrixXd &A,MatrixXd &U, MatrixXd &V,double &epsilon,int &rank,int minRank);
extern "C" int aca_wrapper(double *AA, double *UU, double *VV, int m, int n, double epsilon,int rank);
#endif
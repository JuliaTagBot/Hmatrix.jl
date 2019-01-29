#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <Eigen/Core>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

class Kernel {
public:
  Kernel() {};
  double operator() (double x, double y) const {
    return 1.0/((x-y)*(x-y));
  }
};

// Chebyshev points in [-1, 1]
Vector Chebyshev_points(int n) {
  Vector X(n);
  for (int i=0; i<n; i++)
    X(i) = cos( M_PI*(2.0*i+1.0)/(2.0*n) );
  return X;
}
  
class Interpolation {
public:
  Interpolation(double a_, double b_, int r) {
    a = a_;
    b = b_;
    Cheb = Chebyshev_points(r)*(b-a)/2.0 + Vector::Constant(r, (a+b)/2.0);
  }
  Vector GetChebyshev() const {return Cheb;}
  Matrix ComputeMatrix(const Vector &X) const {
    int N = X.size();
    int r = Cheb.size();
    Matrix P(N, r);
    for (int i=0; i<N; i++)
      for (int j=0; j<r; j++) {
        P(i, j) = 1.0;
        for (int k=0; k<r; k++) {
	 if (k != j)
	   P(i, j) *= (X(i)-Cheb(k))/(Cheb(j)-Cheb(k));
	}
      }
    return P;
  }
private:
  double a;
  double b;
  Vector Cheb;
};

typedef double (*FUN)(double, double);

double Kfunction(double x, double y){
    return 1.0/((x-y)*(x-y));
}

void Compute_lowrank
(FUN K, 
const Vector &X, double xmin, double xmax, 
const Vector &Y, double ymin, double ymax,
const int r, Matrix &U, Matrix &S, Matrix &V) {
  
  Interpolation Ix(xmin, xmax, r);
  Interpolation Iy(ymin, ymax, r);
  U = Ix.ComputeMatrix(X);
  V = Iy.ComputeMatrix(Y);
  Vector chebX = Ix.GetChebyshev();
  Vector chebY = Iy.GetChebyshev();
  for (int i=0; i<r; i++)
    for (int j=0; j<r; j++)
      S(i, j) = K( chebX(i), chebY(j) );
}

extern "C" void bbfmm1D(FUN kfun, double*X, double*Y, double xmin, double xmax, double ymin, double ymax, 
      double *U, double*V, int r, int n1, int n2){
        Vector Vx(n1), Vy(n2);
        for(int i=0;i<n1;i++) Vx(i) = X[i];
        for(int i=0;i<n2;i++) Vy(i) = Y[i];
        Matrix mU(n1,r), mS(r,r), mV(n2,r);
        Compute_lowrank(kfun, Vx, xmin, xmax, Vy, ymin, ymax, r, mU, mS, mV);
        mU = mU*mS;
        for(int i=0;i<mU.rows()*mU.cols();i++){
          U[i] = mU(i);
        }
        for(int i=0;i<mV.rows()*mV.cols();i++){
          V[i] = mV(i);
        }
}

int main(int argc, char *argv[]) {

  int n = 10;
  int r = 3;

  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i], "-r"))
      r = atoi(argv[++i]);
  }

  // random points
  Vector X = Vector::Random(n); // [-1, 1]
  Vector Y = Vector::Random(n) + Vector::Constant(n, 10); // [9, 11] 

  Matrix A(n, n);
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      A(i, j) = Kfunction( X(i), Y(j) );
    }
  }

  Matrix U(n, r), S(r, r), V(n, r);
  Compute_lowrank(Kfunction, X, -1, 1, Y, 9, 11, r, U, S, V);
  
  Matrix E = A - U*S*V.transpose();

  std::cout<<"Hello world!"<<std::endl;
  std::cout<<"X: "<<X.transpose()<<std::endl;
  std::cout<<"Y: "<<Y.transpose()<<std::endl;
  std::cout<<"Matrix: \n"<<A<<std::endl;
  std::cout<<"Error of lowrank: \n"<<E.norm()<<std::endl;
  return 0;
}


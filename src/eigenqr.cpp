#include <iostream>
#include <Eigen/Dense>

namespace eigenqr{
void eigen16x16(float* const eigenvalues, const float* const a, std::size_t n){
	Eigen::MatrixXf ma;
	ma.resize(n, n);
	for(std::size_t i = 0; i < n; i++){
		for(std::size_t j = 0; j < n; j++){
			ma(i, j) = a[i + j * n];
		}
	}
	Eigen::EigenSolver<Eigen::MatrixXf> eigensolver(ma);
	std::cout<<eigensolver.eigenvalues()<<std::endl;
}

}

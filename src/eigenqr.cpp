#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include "eigenqr.hpp"

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
	std::vector<float> sorted_eigenvalues;
	for(std::size_t i = 0; i < n; i++){
		sorted_eigenvalues.push_back(std::abs(eigensolver.eigenvalues()[i].real()));
	}

	std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end(), std::greater<float>());

	for(std::size_t i = 0; i < n; i++){
		eigenvalues[i] = sorted_eigenvalues[i];
	}
}

bool is_real(const float* const a, std::size_t n){
	Eigen::MatrixXf ma;
	ma.resize(n, n);
	for(std::size_t i = 0; i < n; i++){
		for(std::size_t j = 0; j < n; j++){
			ma(i, j) = a[i + j * n];
		}
	}
	Eigen::EigenSolver<Eigen::MatrixXf> eigensolver(ma);
	//std::cout<<<<std::endl;
	float sum = 0.0f;
	for(std::size_t i = 0; i < n; i++){
		sum += std::abs(eigensolver.eigenvalues()[i].imag());
	}
	return (sum == 0.0f);
}

}

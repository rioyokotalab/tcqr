#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cutf/type.hpp>

namespace utils {
template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr){
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			const auto val = cutf::cuda::type::cast<float>(ptr[j * m + i]);
			if(val < 0.0f){
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}

// millisecond
template <class RunFunc>
inline double get_elapsed_time(RunFunc run_func){
	const auto start_clock = std::chrono::system_clock::now();
	run_func();
	const auto end_clock = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() / 1000.0;
}

template <class T>
inline void print_value(const T val, const std::string name){
	std::cout<<std::setw(25)<<name<<" : "<<val<<std::endl;
}
} // namespace utils

#endif /* end of include guard */

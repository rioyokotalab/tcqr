#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <iostream>
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
} // namespace utils

#endif /* end of include guard */

#include <cutf/type.hpp>
#include "tcqr.hpp"

namespace{
constexpr std::size_t warp_size = 32; // 本当はwarpSizeを使いたい

// 2乗和
// sum(ptr[start_id] : ptr[15])
template <class T>
__device__ T get_norm2_16(T* const ptr, const std::size_t start_id, unsigned warp_id){
	T tmp = cutf::cuda::type::cast<T>(0.0f);
	
	// load
	if(start_id <= warp_id && warp_id < 16){
		tmp = ptr[warp_id];
		tmp = tmp * tmp;
	}

	// shfl allreduce
	for(auto mask = (warp_size>>1); mask > 0; mask >>= 1){
		tmp += __shfl_xor_sync(0xffffffff, tmp, mask);
	}
	return tmp;
}
}

template <class Input_t, class Output_t>
void tcqr::qr16x16tc(Output_t *const q, Output_t *const r, const Input_t *const a, const std::size_t m, const std::size_t n){

}

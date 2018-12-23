#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include "tcqr.hpp"

namespace{
constexpr std::size_t warp_size = 32; // 本当はwarpSizeを使いたい
constexpr unsigned fragment_dimension = 16;

// 2乗和
// sum(ptr[start_id] : ptr[15])
template <class T>
__device__ T get_norm2_16(T* const ptr, const std::size_t size, unsigned warp_id){
	T tmp = cutf::cuda::type::cast<T>(0.0f);
	
	// load
	if(warp_id < size){
		tmp = ptr[warp_id];
		tmp = tmp * tmp;
	}

	// shfl allreduce
	for(auto mask = (warp_size>>1); mask > 0; mask >>= 1){
		tmp += __shfl_xor_sync(0xffffffff, tmp, mask);
	}
	return tmp;
}

// Q,R の更新
// UseTC == true -> Input_t = half
template <class Input_t, class Output_t, bool UseTC>
__device__ void update_QR_tc(
		Output_t* const out_q, 
		Output_t* const out_r, 
		const Input_t* const in_q, 
		const Input_t* const in_r, 
		const Input_t* const in_h){
	// コンパイル時に分岐の片方を消してくれることを祈って
	// if constexpr が使えればいいのに
	if (UseTC == true){
		constexpr unsigned fragment_dimension = 16;
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, fragment_dimension, fragment_dimension, fragment_dimension, half, nvcuda::wmma::col_major> in_h_fragment;
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, fragment_dimension, fragment_dimension, fragment_dimension, half, nvcuda::wmma::col_major> in_q_fragment;
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, fragment_dimension, fragment_dimension, fragment_dimension, half, nvcuda::wmma::col_major> in_r_fragment;
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, fragment_dimension, fragment_dimension, fragment_dimension, Output_t> out_q_fragment;
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, fragment_dimension, fragment_dimension, fragment_dimension, Output_t> out_r_fragment;

		nvcuda::wmma::fill_fragment(out_q_fragment, cutf::cuda::type::cast<Output_t>(0.0f));
		nvcuda::wmma::fill_fragment(out_r_fragment, cutf::cuda::type::cast<Output_t>(0.0f));

		nvcuda::wmma::load_matrix_sync(in_h_fragment, in_h, fragment_dimension);
		nvcuda::wmma::load_matrix_sync(in_q_fragment, in_q, fragment_dimension);
		nvcuda::wmma::load_matrix_sync(in_r_fragment, in_r, fragment_dimension);

		nvcuda::wmma::mma_sync(out_q_fragment, in_h_fragment, in_q_fragment, out_q_fragment);
		nvcuda::wmma::mma_sync(out_r_fragment, in_h_fragment, in_r_fragment, out_r_fragment);

		nvcuda::wmma::store_matrix_sync(out_q, out_q_fragment, fragment_dimension, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::store_matrix_sync(out_r, out_r_fragment, fragment_dimension, nvcuda::wmma::mem_col_major);
	}
}
} // noname namespace

template <class Input_t, class Output_t>
void tcqr::qr16x16tc(Output_t *const q, Output_t *const r, const Input_t *const a, const std::size_t m, const std::size_t n){

}

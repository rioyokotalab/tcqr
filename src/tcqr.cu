#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include "tcqr.hpp"
#include "utils.hpp"

#define DEBUG

namespace{
constexpr std::size_t warp_size = 32; // 本当はwarpSizeを使いたい
constexpr unsigned fragment_dimension = 16;

template <class Func>
__device__ void debug_func(unsigned warp_id,	Func run_func){
#ifdef DEBUG
	if(warp_id == 0){
		run_func();
	}
#endif
}

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

// 結合アクセセスを意識
template <class T, class S>
__device__ void copy_16x16(T* const dest_ptr, const S* const src_ptr, unsigned warp_id){
#pragma unroll 
	for(unsigned i = 0; i < fragment_dimension * fragment_dimension / warp_size; i++){
		dest_ptr[warp_size * i + warp_id] = cutf::cuda::type::cast<T>(src_ptr[warp_size * i + warp_id]);
	}
}
template <class T, class S>
__device__ void copy_16x16(T* const dest_ptr, const S* const src_ptr, std::size_t m, std::size_t n, unsigned warp_id){
#pragma unroll 
	for(unsigned i = 0; i < fragment_dimension * fragment_dimension / warp_size; i++){
		const auto index = warp_size * i + warp_id;
		const auto x = index / fragment_dimension;
		const auto y = index % fragment_dimension;
		auto val = cutf::cuda::type::cast<T>(0.0f);
		if(x < m && y < n)
			val = cutf::cuda::type::cast<T>(src_ptr[x * m + y]);;

		dest_ptr[index] = val;
	}
}
// TODO : 結合アクセス
template <class T, class S>
__device__ void copy_16x16(T* const dest_ptr, std::size_t m, std::size_t n, const S* const src_ptr, unsigned warp_id){
#pragma unroll 
	for(unsigned i = 0; i < fragment_dimension * fragment_dimension / warp_size; i++){
		const auto index = warp_size * i + warp_id;
		const auto x = index / fragment_dimension;
		const auto y = index % fragment_dimension;
		auto val = cutf::cuda::type::cast<T>(0.0f);
		if(x < m && y < n)
			dest_ptr[x * m + y] = cutf::cuda::type::cast<T>(src_ptr[index]);
	}
}
template <class T, class S>
__device__ void copy_16(T* const dest_ptr, const S* const src_ptr, unsigned warp_id){
	if(warp_id < fragment_dimension){
		dest_ptr[warp_id] = cutf::cuda::type::cast<T>(src_ptr[warp_id]);
	}
}

template <class T>
__device__ void make_identity_matrix(T* const dest_ptr, std::size_t m, unsigned warp_id){
	for(unsigned i = 0; i < fragment_dimension * fragment_dimension / warp_size; i++){
		const auto index = warp_size * i + warp_id;
		if(index % (fragment_dimension + 1) == 0) dest_ptr[index] = cutf::cuda::type::cast<T>(1.0f);
		else dest_ptr[index] = cutf::cuda::type::cast<T>(0.0f);
	}
}

// 結合アクセセスを意識
template <class T, class S>
__device__ void make_h(T* const h, const S* const u, const S norm_u2, unsigned warp_id){
#pragma unroll 
	for(unsigned i = 0; i < fragment_dimension * fragment_dimension / warp_size; i++){
		const auto index = warp_size * i + warp_id;
		const auto x = index / fragment_dimension;
		const auto y = index % fragment_dimension;

		// 単位行列生成は make_identity_matrix関数を用いない
		// メモリアクセスを減らせる
		T val;
		if(index % (fragment_dimension + 1) == 0) val = cutf::cuda::type::cast<T>(1.0f);
		else val = cutf::cuda::type::cast<T>(0.0f);

		val -= cutf::cuda::type::cast<T>(2.0f) * u[x] * u[y] * cutf::cuda::math::rcp(norm_u2);
		h[index] = val;
	}
}

// Q,R の更新
template <class Input_t, class Output_t>
__device__ void update_QR_tc(
		Output_t* const out_q, 
		Output_t* const out_r, 
		const Input_t* const in_q, 
		const Input_t* const in_r, 
		const Input_t* const in_h){
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

// tcqr
// 入出力はShared memoryで
// out_q/out_rの初期化は関数の手前で行っておくこと
// out_q <- Identity matrix
// out_r <- a
template <class Input_t, class Output_t, class Norm_t>
__device__ void qr16x16tc_core(Output_t* const out_q, Output_t* const out_r, const std::size_t m, const std::size_t n, unsigned warp_id);
template <>
__device__ void qr16x16tc_core<half, half, half>(half* const out_q, half* const out_r, const std::size_t m, const std::size_t n, unsigned warp_id){
	__shared__ half h[fragment_dimension * fragment_dimension];
	__shared__ half u[fragment_dimension];

	for(std::size_t k = 0; k < n; k++){
		copy_16(u, out_r + fragment_dimension * k, warp_id);
		if(warp_id < k){
			u[warp_id] = cutf::cuda::type::cast<half>(0.0f);
		}

		const auto norm_u = cutf::cuda::math::sqrt(get_norm2_16(u, m, warp_id));
		if(warp_id == k){
			u[warp_id] += norm_u * cutf::cuda::math::sign(u[warp_id]);
		}
		
		const auto norm_u2 = get_norm2_16(u, m, warp_id);
		make_h(h, u, norm_u2, warp_id);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
		update_QR_tc<half, half>(out_q, out_r, out_q, out_r, h);
#endif
	}
}

// kernel
template <class Input_t, class Output_t, class Norm_t>
__global__ void qr16x16tc_kernel(Output_t* const q, Output_t* const r, const Input_t* const a, const std::size_t m, const std::size_t n);
template <>
__global__ void qr16x16tc_kernel<half, half, half>(half* const q, half* const r, const half* const a, const std::size_t m, const std::size_t n){
	const auto warp_id = threadIdx.x & 0xff;
	__shared__ half q_shared[fragment_dimension * fragment_dimension];
	__shared__ half r_shared[fragment_dimension * fragment_dimension];

	copy_16x16<half, half>(r_shared, a, m, n, warp_id);
	make_identity_matrix(q_shared, m, warp_id);

	qr16x16tc_core<half, half, half>(q_shared, r_shared, m, n, warp_id);

	copy_16x16<half, half>(r, m, n, r_shared, warp_id);
	copy_16x16<half, half>(q, m, m, q_shared, warp_id);
}
} // noname namespace

template <class Input_t, class Output_t, class Norm_t, bool UseTC>
void tcqr::qr16x16(Output_t *const q, Output_t *const r, const Input_t *const a, const std::size_t m, const std::size_t n){
	if(UseTC){
		qr16x16tc_kernel<Input_t, Output_t, Norm_t><<<1, warp_size>>>(q, r, a, m, n);
	}
}

template void tcqr::qr16x16<half, half, half, true>(half *const, half *const, const half *const, const std::size_t, const std::size_t);

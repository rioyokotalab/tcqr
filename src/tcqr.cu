#include <type_traits>
#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include "tcqr.hpp"
#include "utils.hpp"

//#define DEBUG

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
		auto val = cutf::cuda::type::cast<S>(0.0f);
		if(x < n && y < m)
			val = cutf::cuda::type::cast<S>(src_ptr[x * m + y]);;

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
		if(x < n && y < m)
			dest_ptr[x * m + y] = cutf::cuda::type::cast<S>(src_ptr[index]);
	}
}
template <class T, class S>
__device__ void copy_16(T* const dest_ptr, const S* const src_ptr, unsigned warp_id){
	if(warp_id < fragment_dimension){
		dest_ptr[warp_id] = cutf::cuda::type::cast<T>(src_ptr[warp_id]);
	}
}

// 行列積
template <class T>
__device__ void matmul_16x16_TN(T* const c, const T* const a, const T* const b, unsigned warp_id){
	const auto start_i = (warp_id & 0x1) * (fragment_dimension/2);
	const auto j = (warp_id >> 1);

	for(std::size_t i = start_i; i < fragment_dimension / 2 + start_i; i++){
		T sum = 0.0f;
		for(std::size_t k = 0; k < fragment_dimension; k++){
			sum += a[fragment_dimension * i + k] * b[fragment_dimension * j + k];
		}
		c[fragment_dimension * j + i] = sum;
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
template <class T, bool UseTC>
__device__ void update_QR_homogeneous(
		T* const out_q, 
		T* const out_r, 
		const T* const in_q, 
		const T* const in_r, 
		const T* const in_h,
		unsigned warp_id){
	// TODO : hの再利用
	if(UseTC){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
		update_QR_tc<half, half>(out_q, out_r, in_q, in_r, in_h);
#endif
	}else{
		matmul_16x16_TN(out_q, in_h, in_q, warp_id);
		matmul_16x16_TN(out_r, in_h, in_r, warp_id);
	}
}

// tcqr
// 入出力はShared memoryで
// out_q/out_rの初期化は関数の手前で行っておくこと
// out_q <- Identity matrix
// out_r <- a
template <class T, class Norm_t, bool UseTC>
__device__ void qr16x16_homogeneous_core(T* const out_q, T* const out_r, const std::size_t m, const std::size_t n, unsigned warp_id){
	__shared__ T h[fragment_dimension * fragment_dimension];
	__shared__ T u[fragment_dimension];

	for(std::size_t k = 0; k < n; k++){
		debug_func(warp_id,
				[&k](){printf(
					"//---------------------\n"
					"// k = %lu\n"
					"//---------------------\n"
					, k);});
		debug_func(warp_id,
				[&m, &n, &out_r](){utils::print_matrix(out_r, 16, 16, "r");});
		debug_func(warp_id,
				[&m, &out_q](){utils::print_matrix(out_q, 16, 16, "q");});

		copy_16(u, out_r + fragment_dimension * k, warp_id);
		if(warp_id < k){
			u[warp_id] = cutf::cuda::type::cast<T>(0.0f);
		}
		debug_func(warp_id,
				[](){utils::print_matrix(u, 1, 16, "u");});

		const auto norm_u = cutf::cuda::math::sqrt(get_norm2_16(u, m, warp_id));
		if(warp_id == k){
			u[warp_id] += norm_u * cutf::cuda::math::sign(u[warp_id]);
		}
		debug_func(warp_id,
				[](){utils::print_matrix(u, 1, 16, "u+");});

		const auto norm_u2 = get_norm2_16(u, m, warp_id);
		make_h(h, u, norm_u2, warp_id);
		update_QR_homogeneous<T, UseTC>(out_q, out_r, out_q, out_r, h, warp_id);
	}
}

// kernel
template <class T, class Norm_t, bool UseTC>
__global__ void qr16x16_homogeneous_kernel(T* const q, T* const r, const T* const a, const std::size_t m, const std::size_t n){
	const auto warp_id = threadIdx.x & 0xff;
	__shared__ T q_shared[fragment_dimension * fragment_dimension];
	__shared__ T r_shared[fragment_dimension * fragment_dimension];

	copy_16x16<T, T>(r_shared, a, m, n, warp_id);
	make_identity_matrix(q_shared, m, warp_id);

	qr16x16_homogeneous_core<T, Norm_t, UseTC>(q_shared, r_shared, m, n, warp_id);

	copy_16x16<T, T>(r, m, n, r_shared, warp_id);
	copy_16x16<T, T>(q, m, m, q_shared, warp_id);
}
} // noname namespace

template <class Input_t, class Output_t, class Norm_t, bool UseTC>
void tcqr::qr16x16(Output_t *const q, Output_t *const r, const Input_t *const a, const std::size_t m, const std::size_t n){
	/*if(UseTC){
	  qr16x16tc_kernel<Input_t, Output_t, Norm_t><<<1, warp_size>>>(q, r, a, m, n);
	  }else{
	  qr16x16_kernel<Input_t, Output_t, Norm_t><<<1, warp_size>>>(q, r, a, m, n);
	  }*/
	if(std::is_same<Output_t, Input_t>::value == true){
		qr16x16_homogeneous_kernel<Input_t, Norm_t, UseTC><<<1, warp_size>>>(q, r, a, m, n);
	}else{

	}
}

template void tcqr::qr16x16<half, half, half, true>(half *const, half *const, const half *const, const std::size_t, const std::size_t);

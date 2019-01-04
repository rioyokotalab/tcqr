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
// 内部計算をS型で行い，S型で返す
// sum(ptr[start_id] : ptr[15])
template <class T, class S>
__device__ S get_norm2_16(T* const ptr, const std::size_t size, unsigned warp_id){
	auto tmp = cutf::cuda::type::cast<S>(0.0f);
	
	// load
	if(warp_id < size){
		tmp = cutf::cuda::type::cast<S>(ptr[warp_id]);
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

// Globalメモリアクセスを結合アクセスにすると遅くなる気がする．
// 要素位置(x, y)の計算などで差がつくのかも?
template <class T, class S>
__device__ void copy_16x16_T(T* const dest_ptr, std::size_t m, std::size_t n, const S* const src_ptr, unsigned warp_id){
#pragma unroll
	for(unsigned i = 0; i < fragment_dimension * fragment_dimension / warp_size; i++){
		const auto index = warp_size * i + warp_id;
		const auto x = index / fragment_dimension;
		const auto y = index % fragment_dimension;
		if(x < n && y < m)
			dest_ptr[m * y + x] = cutf::cuda::type::cast<S>(src_ptr[index]);
	}
}

template <class T, class S>
__device__ void copy_16(T* const dest_ptr, const S* const src_ptr, unsigned warp_id){
	if(warp_id < fragment_dimension){
		dest_ptr[warp_id] = cutf::cuda::type::cast<T>(src_ptr[warp_id]);
	}
}

// 行列積
// Bが対称行列の場合，C <- A * BはC <- A^T * Bと同値
// 連続メモリアクセスのためTNで計算する
template <class T>
__device__ void matmul_16x16_TN(T* const c, const T* const a, const T* const b, unsigned warp_id){
	/* 行列Cを1ワープで計算する
	 * スレッドによる分割方法は
	 * C(列優先) = 
	 * -------------------- -
	 * |   |   | ... |    | ^
	 * | 0 | 2 | ... | 30 | |
	 * |   |   | ... |    | |
	 * -------------------- 16
	 * |   |   | ... |    | |
	 * | 1 | 3 | ... | 31 | |
	 * |   |   | ... |    | v
	 * -------------------- -
	 * <--------16-------->
	 * の様に分割する．
	 * (start_i, j)は各スレッドの書き込み先の
	 * 先頭の要素
	 */
	// (x % 2) <-> (x & 0x1)
	const auto start_i = (warp_id & 0x1) * (fragment_dimension/2);
	// (x / 2) <-> (x >> 1)
	const auto j = (warp_id >> 1);
	T sums[fragment_dimension/2];

	for(std::size_t i = start_i; i < fragment_dimension / 2 + start_i; i++){
		auto sum = cutf::cuda::type::cast<T>(0.0f);
		for(std::size_t k = 0; k < fragment_dimension; k++){
			sum += a[fragment_dimension * i + k] * b[fragment_dimension * j + k];
		}
		sums[i - start_i] = sum;
	}
	__syncthreads();

	// 一度バッファ(レジスタ)に貯めてからメモリに書き込み
	for(std::size_t i = start_i; i < fragment_dimension / 2 + start_i; i++){
		c[fragment_dimension * j + i] = sums[i - start_i];
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

		val -= cutf::cuda::type::cast<T>(2.0f) * cutf::cuda::type::cast<T>(u[x] * u[y] * cutf::cuda::math::rcp(norm_u2));
		h[index] = val;
	}
}

// Q,R の更新
template <class Input_t, class Output_t>
__device__ void update_qr_tc(
		Output_t* const out_q, 
		Output_t* const out_r, 
		const Input_t* const in_q, 
		const Input_t* const in_r, 
		const Input_t* const in_h){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
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
#endif
}

// 非TCQ,R更新関数
template <class T, bool UseTC>
__device__ void update_qr(T* const out_q, T* const out_r, const T* const in_q, const T* const in_r, const T* const in_h,unsigned warp_id){
	// TODO : hの再利用
	matmul_16x16_TN(out_q, in_h, in_q, warp_id);
	matmul_16x16_TN(out_r, in_h, in_r, warp_id);
}
template <>
__device__ void update_qr<half, true>(half* const out_q, half* const out_r, const half* const in_q, const half* const in_r, const half* const in_h,unsigned warp_id){
	update_qr_tc<half, half>(out_q, out_r, in_q, in_r, in_h);
}

// tcqr
// 入出力はShared memoryで
// out_q/out_rの初期化は関数の手前で行っておくこと
// out_q <- Identity matrix
// out_r <- a
template <class T, class Norm_t, bool UseTC>
__device__ void qr16x16_core(T* const out_q, T* const out_r, 
		T* const h, T* const u,
		const std::size_t m, const std::size_t n, unsigned warp_id){
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
				[&u](){utils::print_matrix(u, 1, 16, "u");});

		const auto norm_u = cutf::cuda::math::sqrt(cutf::cuda::type::cast<T>(get_norm2_16<T, Norm_t>(u, m, warp_id)));
		if(warp_id == k){
			u[warp_id] += norm_u * cutf::cuda::math::sign(u[warp_id]);
		}
		debug_func(warp_id,
				[&u](){utils::print_matrix(u, 1, 16, "u+");});

		const auto norm_u2 = cutf::cuda::type::cast<T>(get_norm2_16<T, Norm_t>(u, m, warp_id));
		make_h(h, u, norm_u2, warp_id);
		update_qr<T, UseTC>(out_q, out_r, out_q, out_r, h, warp_id);
	}
}
__device__ void qr16x16_f32tc_core(
		float * const q_f32, float* const r_f32,
		half* const q_f16, half* const r_f16,
		float* const u_f32, half* const h_f16,
		const std::size_t m, const std::size_t n, unsigned warp_id){
	for(std::size_t k = 0; k < n; k++){
		debug_func(warp_id,
				[&k](){printf(
					"//---------------------\n"
					"// k = %lu\n"
					"//---------------------\n"
					, k);});
		debug_func(warp_id,
				[&q_f32](){utils::print_matrix(q_f32, 16, 16, "q");});
		debug_func(warp_id,
				[&r_f32](){utils::print_matrix(r_f32, 16, 16, "r");});

		copy_16(u_f32, r_f32 + fragment_dimension * k, warp_id);
		if(warp_id < k){
			u_f32[warp_id] = 0.0f;
		}
		debug_func(warp_id,
				[&u_f32](){utils::print_matrix(u_f32, 1, 16, "u");});

		const auto norm_u = cutf::cuda::math::sqrt(get_norm2_16<float, float>(u_f32, m, warp_id));
		if(warp_id == k){
			u_f32[warp_id] += norm_u * cutf::cuda::math::sign(u_f32[warp_id]);
		}
		debug_func(warp_id,
				[&u_f32](){utils::print_matrix(u_f32, 1, 16, "u+");});

		const auto norm_u2 = get_norm2_16<float, float>(u_f32, m, warp_id);
		make_h(h_f16, u_f32, norm_u2, warp_id);
		// q,r の型変換
		copy_16x16<half, float>(q_f16, q_f32, warp_id);
		copy_16x16<half, float>(r_f16, r_f32, warp_id);

		update_qr_tc<half, float>(q_f32, r_f32, q_f16, r_f16, h_f16);
	}
}

// kernel
template <class T, class Norm_t, bool UseTC>
__global__ void qr16x16_kernel(T* const q, T* const r, const T* const a, const std::size_t m, const std::size_t n){
	// (x % 32) <-> (x & 0x1f)
	const auto warp_id = threadIdx.x & 0x1f;
	__shared__ T q_shared[fragment_dimension * fragment_dimension];
	__shared__ T r_shared[fragment_dimension * fragment_dimension];
	__shared__ T h[fragment_dimension * fragment_dimension];
	__shared__ T u[fragment_dimension];


	copy_16x16<T, T>(r_shared, a, m, n, warp_id);
	make_identity_matrix(q_shared, m, warp_id);

	qr16x16_core<T, Norm_t, UseTC>(q_shared, r_shared,
			h, u,
		   	m, n, warp_id);

	copy_16x16<T, T>(r, m, n, r_shared, warp_id);
	copy_16x16_T<T, T>(q, m, m, q_shared, warp_id);
}

// 単精度入出力TC使用
__global__ void qr16x16_f32tc_kernel(float* const q, float* const r, const float* const a, const std::size_t m, const std::size_t n){
	// (x % 32) <-> (x & 0x1f)
	const auto warp_id = threadIdx.x & 0x1f;
	__shared__ float q_shared_f32[fragment_dimension * fragment_dimension];
	__shared__ float r_shared_f32[fragment_dimension * fragment_dimension];
	__shared__ half q_shared_f16[fragment_dimension * fragment_dimension];
	__shared__ half r_shared_f16[fragment_dimension * fragment_dimension];
	__shared__ half h_shared[fragment_dimension * fragment_dimension];
	__shared__ float u_shared[fragment_dimension];

	copy_16x16(r_shared_f32, a, m, n, warp_id);
	make_identity_matrix(q_shared_f32, m, warp_id);

	qr16x16_f32tc_core(q_shared_f32, r_shared_f32,
			q_shared_f16, r_shared_f16,
			u_shared, h_shared,
			m, n, warp_id);


	copy_16x16(r, m, n, r_shared_f32, warp_id);
	copy_16x16_T(q, m, m, q_shared_f32, warp_id);
}
} // noname namespace

// if constexpr が使えるようになったら書き直せ!!!!
template <class Input_t, class Output_t, class Norm_t, bool UseTC>
void tcqr::qr16x16(Output_t *const q, Output_t *const r, const Input_t *const a, const std::size_t m, const std::size_t n){
	qr16x16_kernel<Output_t, Norm_t, UseTC><<<1, warp_size>>>(q, r, a, m, n);
}
template <> void tcqr::qr16x16<float, float, float, true>(float *const q, float *const r, const float *const a, const std::size_t m, const std::size_t n){qr16x16_f32tc_kernel<<<1, warp_size>>>(q, r, a, m, n);};
template void tcqr::qr16x16<half, half, half, true>(half *const, half *const, const half *const, const std::size_t, const std::size_t);
template void tcqr::qr16x16<half, half, float, true>(half *const, half *const, const half *const, const std::size_t, const std::size_t);
template void tcqr::qr16x16<half, half, half, false>(half *const, half *const, const half *const, const std::size_t, const std::size_t);
template void tcqr::qr16x16<half, half, float, false>(half *const, half *const, const half *const, const std::size_t, const std::size_t);
template void tcqr::qr16x16<float, float, float, false>(float *const, float *const, const float *const, const std::size_t, const std::size_t);
//template <> void tcqr::qr16x16<double, double, double, false>(double *const q, double *const r, const double *const a, const std::size_t m, const std::size_t n){qr16x16_heterogeneous_kernel<double, double, double, true><<<1, warp_size>>>(q, r, a, m, n);};

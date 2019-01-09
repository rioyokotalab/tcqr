#include <random>
#include <iostream>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include "test.hpp"
#include "utils.hpp"
#include "tcqr.hpp"

// #define PRINT_MATRIX

namespace{
template <class T>std::string get_type_name();
template <> std::string get_type_name<float>(){return "float";};
template <> std::string get_type_name<half>(){return "half";};

// 副作用があるっぽく見せるために適当なポインタ引数を取るようにする
// nvccの最適化で消されないようにするため
__global__ void tc_warning_kernel(void* p){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
	printf("This device cannot execute code using TensorCore\n");
#endif
}
void tc_warning(){
	tc_warning_kernel<<<1, 1>>>(nullptr);
}
template <class T>
cudaDataType_t get_cuda_data_t();
template <> cudaDataType_t get_cuda_data_t<half>(){return CUDA_R_16F;}
template <> cudaDataType_t get_cuda_data_t<float>(){return CUDA_R_32F;}
}

template <class T, class Norm_t, bool UseTC, std::size_t test_count>
void test::time::qr(const std::size_t m, const std::size_t n, const float* const a){
	if(UseTC)
		tc_warning();
	auto d_matrix_a = cutf::cuda::memory::get_device_unique_ptr<T>(m * n);
	auto d_matrix_r = cutf::cuda::memory::get_device_unique_ptr<T>(m * n);
	auto d_matrix_q = cutf::cuda::memory::get_device_unique_ptr<T>(m * m);
	auto d_matrix_qr = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);

	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<T>(m * n);
	auto h_matrix_qr = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);

	// print type information{{{
	utils::print_value(test_count, "Test count");
	utils::print_value(std::to_string(m) + " x " + std::to_string(n), "Matrix size");
	utils::print_value(get_type_name<T>(), "Input/Output type");
	utils::print_value(get_type_name<Norm_t>(), "Norm type");
	utils::print_value((UseTC ? "true" : "false"), "Use TC?");
	// }}}

	// copy
	for(std::size_t i = 0; i < m * n; i++){
		h_matrix_a.get()[i] = cutf::cuda::type::cast<T>(a[i]);
	}

	cutf::cuda::memory::copy(d_matrix_a.get(), h_matrix_a.get(), m * n);
	auto elapsed_time = utils::get_elapsed_time(
			[&d_matrix_q, &d_matrix_r, &d_matrix_a, &m, &n](){
			for(std::size_t c = 0; c < test_count; c++)
				tcqr::qr16x16<T, Norm_t, UseTC>(d_matrix_q.get(), d_matrix_r.get(), d_matrix_a.get(), m, n);
			cudaDeviceSynchronize();
			});
	utils::print_value(elapsed_time / test_count, "Elapsed time", "ms");
	utils::print_value(test_count * 16 * 16 * 16 * 2 * 2 * (n-1) / elapsed_time * 1000.0 / 1000000000.0, "", "GFLOPS");


	// 検証
	const float one = 1.0f;
	const float zero = 0.0f;
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	cublasGemmEx(
			*cublas.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			m, n, m,
			&one,
			d_matrix_q.get(), get_cuda_data_t<T>(), m,
			d_matrix_r.get(), get_cuda_data_t<T>(), m,
			&zero,
			d_matrix_qr.get(), get_cuda_data_t<float>(), m,
			get_cuda_data_t<float>(), CUBLAS_GEMM_DEFAULT
			);
	cutf::cuda::memory::copy(h_matrix_qr.get(), d_matrix_qr.get(), m * n);

#ifdef PRINT_MATRIX
	auto h_matrix_r = cutf::cuda::memory::get_host_unique_ptr<T>(m * n);
	auto h_matrix_q = cutf::cuda::memory::get_host_unique_ptr<T>(m * m);
	cutf::cuda::memory::copy(h_matrix_q.get(), d_matrix_q.get(), m * m);
	cutf::cuda::memory::copy(h_matrix_r.get(), d_matrix_r.get(), m * n);
	utils::print_matrix(h_matrix_a.get(), m, n, std::string("A").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_q.get(), m, m, std::string("Q").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_r.get(), m, n, std::string("R").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_qr.get(), m, n, std::string("QR").c_str());
#endif

	const auto error = utils::get_error(a, h_matrix_qr.get(), m, n);
	utils::print_value(error , "error");
	std::cout<<std::endl;
}

template <class T, class Norm_t, bool UseTC, std::size_t test_count>
void test::precision::qr(const std::size_t m, const std::size_t n){
	if(UseTC)
		tc_warning();
	auto d_matrix_a = cutf::cuda::memory::get_device_unique_ptr<T>(m * n);
	auto d_matrix_r = cutf::cuda::memory::get_device_unique_ptr<T>(m * n);
	auto d_matrix_q = cutf::cuda::memory::get_device_unique_ptr<T>(m * m);
	auto d_matrix_qr = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);

	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<T>(m * n);
	auto h_matrix_a_f32 = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_matrix_qr = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);

	// print type information{{{
	utils::print_value(test_count, "Test count");
	utils::print_value(std::to_string(m) + " x " + std::to_string(n), "Matrix size");
	utils::print_value(get_type_name<T>(), "Input/Output type");
	utils::print_value(get_type_name<Norm_t>(), "Norm type");
	utils::print_value((UseTC ? "true" : "false"), "Use TC?");
	// }}}

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	float error_sum = 0.0f;
	for(std::size_t i = 0; i < test_count; i++){
		// copy
		for(std::size_t i = 0; i < m * n; i++){
			const auto val = dist(mt);
			h_matrix_a_f32.get()[i] = val;
			h_matrix_a.get()[i] = cutf::cuda::type::cast<T>(val);
		}

		cutf::cuda::memory::copy(d_matrix_a.get(), h_matrix_a.get(), m * n);

		tcqr::qr16x16<T, Norm_t, UseTC>(d_matrix_q.get(), d_matrix_r.get(), d_matrix_a.get(), m, n);

		// 検証
		const float one = 1.0f;
		const float zero = 0.0f;
		cublasGemmEx(
				*cublas.get(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, m,
				&one,
				d_matrix_q.get(), get_cuda_data_t<T>(), m,
				d_matrix_r.get(), get_cuda_data_t<T>(), m,
				&zero,
				d_matrix_qr.get(), get_cuda_data_t<float>(), m,
				get_cuda_data_t<float>(), CUBLAS_GEMM_DEFAULT
				);
		cutf::cuda::memory::copy(h_matrix_qr.get(), d_matrix_qr.get(), m * n);
		const auto error = utils::get_error(h_matrix_a_f32.get(), h_matrix_qr.get(), m, n);
		error_sum += error;
	}
	utils::print_value(error_sum/test_count , "error avg");
	std::cout<<std::endl;
}

template void test::time::qr<half, half, true>(const std::size_t, const std::size_t, const float* const);
template void test::time::qr<half, half, false>(const std::size_t, const std::size_t, const float* const);
template void test::time::qr<half, float, true>(const std::size_t, const std::size_t, const float* const);
template void test::time::qr<half, float, false>(const std::size_t, const std::size_t, const float* const);
template void test::time::qr<float, float, false>(const std::size_t, const std::size_t, const float* const);
template void test::time::qr<float, float, true>(const std::size_t, const std::size_t, const float* const);

template void test::precision::qr<half, half, true>(const std::size_t, const std::size_t);
template void test::precision::qr<half, half, false>(const std::size_t, const std::size_t);
template void test::precision::qr<half, float, true>(const std::size_t, const std::size_t);
template void test::precision::qr<half, float, false>(const std::size_t, const std::size_t);
template void test::precision::qr<float, float, false>(const std::size_t, const std::size_t);
template void test::precision::qr<float, float, true>(const std::size_t, const std::size_t);

template <class T, class Norm_t, bool UseTC, std::size_t test_count>
void test::time::eigen(const std::size_t n, const float* const a){
	//eigen_eigen(a, n);return;
	if(UseTC)
		tc_warning();
	auto d_matrix_a = cutf::cuda::memory::get_device_unique_ptr<T>(n * n);
	auto d_eigenvalues = cutf::cuda::memory::get_device_unique_ptr<T>(n);
	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<T>(n * n);
	auto h_eigenvalues = cutf::cuda::memory::get_host_unique_ptr<T>(n);

	// print type information{{{
	utils::print_value(test_count, "Test count");
	utils::print_value(std::to_string(n) + " x " + std::to_string(n), "Matrix size");
	utils::print_value(get_type_name<T>(), "Input type");
	utils::print_value(get_type_name<Norm_t>(), "Norm type");
	utils::print_value((UseTC ? "true" : "false"), "Use TC?");
#ifdef PRINT_MATRIX
	utils::print_matrix(a, n, n, "a");
#endif
	// }}}

	// copy
	for(std::size_t i = 0; i < n * n; i++){
		h_matrix_a.get()[i] = cutf::cuda::type::cast<T>(a[i]);
	}
	cutf::cuda::memory::copy(d_matrix_a.get(), h_matrix_a.get(), n * n);
	auto elapsed_time = utils::get_elapsed_time(
			[&d_eigenvalues, &d_matrix_a, &n](){
			for(std::size_t c = 0; c < test_count; c++)
			tcqr::eigen16x16<T, Norm_t, UseTC>(d_eigenvalues.get(), d_matrix_a.get(), n);
			cudaDeviceSynchronize();
			});
	utils::print_value(elapsed_time / test_count, "Elapsed time", "ms");

	cutf::cuda::memory::copy(h_eigenvalues.get(), d_eigenvalues.get(), n);
	utils::print_matrix(h_eigenvalues.get(), 1, n, "Eigenvalue");
	std::cout<<std::endl;
}

template void test::time::eigen<half, half, false>(const std::size_t, const float* const);
template void test::time::eigen<half, half, true>(const std::size_t, const float* const);
template void test::time::eigen<half, float, false>(const std::size_t, const float* const);
template void test::time::eigen<half, float, true>(const std::size_t, const float* const);
template void test::time::eigen<float, float, false>(const std::size_t, const float* const);
template void test::time::eigen<float, float, true>(const std::size_t, const float* const);

template <class T, class Norm_t, bool UseTC, std::size_t test_count>
void test::time::qr_batched(const std::size_t m, const std::size_t n, const std::size_t batch_size){
	if(UseTC)
		tc_warning();
	auto d_matrix_a_array = cutf::cuda::memory::get_device_unique_ptr<T*>(batch_size);
	auto d_matrix_q_array = cutf::cuda::memory::get_device_unique_ptr<T*>(batch_size);
	auto d_matrix_r_array = cutf::cuda::memory::get_device_unique_ptr<T*>(batch_size);

	auto h_matrix_a_array = cutf::cuda::memory::get_host_unique_ptr<T*>(batch_size);
	auto h_matrix_q_array = cutf::cuda::memory::get_host_unique_ptr<T*>(batch_size);
	auto h_matrix_r_array = cutf::cuda::memory::get_host_unique_ptr<T*>(batch_size);

	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<T>(m * n);
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for(std::size_t b = 0; b < batch_size; b++){
		T* a_ptr;
		T* q_ptr;
		T* r_ptr;
		cudaMalloc((void**)&a_ptr, sizeof(T) * m * n);
		cudaMalloc((void**)&r_ptr, sizeof(T) * m * n);
		cudaMalloc((void**)&q_ptr, sizeof(T) * m * m);

		for(std::size_t i = 0; i < m * n; i++){
			h_matrix_a.get()[i] = cutf::cuda::type::cast<T>(dist(mt));
		}
		cutf::cuda::memory::copy(a_ptr, h_matrix_a.get(), m * n);

		h_matrix_a_array.get()[b] = a_ptr;
		h_matrix_q_array.get()[b] = q_ptr;
		h_matrix_r_array.get()[b] = r_ptr;
	}
	utils::print_value(test_count, "Test count");
	utils::print_value(batch_size, "Batch size");
	utils::print_value(std::to_string(n) + " x " + std::to_string(n), "Matrix size");
	utils::print_value(get_type_name<T>(), "Input type");
	utils::print_value(get_type_name<Norm_t>(), "Norm type");
	utils::print_value((UseTC ? "true" : "false"), "Use TC?");

	cutf::cuda::memory::copy(d_matrix_a_array.get(), h_matrix_a_array.get(), batch_size);
	cutf::cuda::memory::copy(d_matrix_q_array.get(), h_matrix_q_array.get(), batch_size);
	cutf::cuda::memory::copy(d_matrix_r_array.get(), h_matrix_r_array.get(), batch_size);
	auto elapsed_time = utils::get_elapsed_time(
			[&d_matrix_q_array, &d_matrix_r_array, &d_matrix_a_array, &m, &n, &batch_size](){
			for(std::size_t c = 0; c < test_count; c++)
				tcqr::qr16x16_batched<T, Norm_t, UseTC>(d_matrix_q_array.get(), d_matrix_r_array.get(), d_matrix_a_array.get(), m, n, batch_size);
			cudaDeviceSynchronize();
			});
	utils::print_value(elapsed_time / test_count, "Elapsed time", "ms");
	utils::print_value(batch_size * test_count * 16 * 16 * 16 * 2 * 2 * (n-1) / elapsed_time * 1000.0 / 1000000000.0, "", "GFLOPS");
	std::cout<<std::endl;

	for(std::size_t b = 0; b < batch_size; b++){
		cudaFree(h_matrix_a_array.get()[b]);
		cudaFree(h_matrix_q_array.get()[b]);
		cudaFree(h_matrix_r_array.get()[b]);
	}
}
template void test::time::qr_batched<half, half, true>(const std::size_t, const std::size_t ,const std::size_t);
template void test::time::qr_batched<half, half, false>(const std::size_t, const std::size_t ,const std::size_t);
template void test::time::qr_batched<half, float, true>(const std::size_t, const std::size_t ,const std::size_t);
template void test::time::qr_batched<half, float, false>(const std::size_t, const std::size_t ,const std::size_t);
template void test::time::qr_batched<float, float, false>(const std::size_t, const std::size_t ,const std::size_t);
template void test::time::qr_batched<float, float, true>(const std::size_t, const std::size_t ,const std::size_t);

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
}

template <class Input_t, class Output_t, class Norm_t, bool UseTC, std::size_t test_count>
void test::qr(const std::size_t m, const std::size_t n, const float* const a){
	if(UseTC)
		tc_warning();
	auto d_matrix_a = cutf::cuda::memory::get_device_unique_ptr<Input_t>(m * n);
	auto d_matrix_r = cutf::cuda::memory::get_device_unique_ptr<Output_t>(m * n);
	auto d_matrix_q = cutf::cuda::memory::get_device_unique_ptr<Output_t>(m * m);
	auto d_matrix_qr = cutf::cuda::memory::get_device_unique_ptr<Output_t>(m * n);

	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<Input_t>(m * n);
	auto h_matrix_r = cutf::cuda::memory::get_host_unique_ptr<Input_t>(m * n);
	auto h_matrix_q = cutf::cuda::memory::get_host_unique_ptr<Input_t>(m * m);
	auto h_matrix_qr = cutf::cuda::memory::get_host_unique_ptr<Input_t>(m * n);

	// print type information{{{
	utils::print_value(test_count, "Test count");
	utils::print_value(std::to_string(m) + " x " + std::to_string(n), "Matrix size");
	utils::print_value(get_type_name<Input_t>(), "Input type");
	utils::print_value(get_type_name<Output_t>(), "Output type");
	utils::print_value(get_type_name<Norm_t>(), "Norm type");
	utils::print_value((UseTC ? "true" : "false"), "Use TC?");
	// }}}

	// copy
	for(std::size_t i = 0; i < m * n; i++){
		h_matrix_a.get()[i] = cutf::cuda::type::cast<Input_t>(a[i]);
	}

	cutf::cuda::memory::copy(d_matrix_a.get(), h_matrix_a.get(), m * n);
	auto elapsed_time = utils::get_elapsed_time(
			[&d_matrix_q, &d_matrix_r, &d_matrix_a, &m, &n](){
			for(std::size_t c = 0; c < test_count; c++)
				tcqr::qr16x16<Input_t, Output_t, Norm_t, UseTC>(d_matrix_q.get(), d_matrix_r.get(), d_matrix_a.get(), m, n);
			cudaDeviceSynchronize();
			});
	utils::print_value(elapsed_time / test_count, "Elapsed time", "ms");


	// 検証
	Output_t one = cutf::cuda::type::cast<Output_t>(1.0f);
	Output_t zero = cutf::cuda::type::cast<Output_t>(0.0f);
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			m, n, m,
			&one,
			d_matrix_q.get(), m,
			d_matrix_r.get(), m,
			&zero,
			d_matrix_qr.get(), m
			);
	cutf::cuda::memory::copy(h_matrix_qr.get(), d_matrix_qr.get(), m * n);

#ifdef PRINT_MATRIX
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

	const auto error = utils::get_error(h_matrix_a.get(), h_matrix_qr.get(), m, n);
	utils::print_value(error , "error");
	std::cout<<std::endl;
}

template void test::qr<half, half, half, true>(const std::size_t, const std::size_t, const float* const);
template void test::qr<half, half, half, false>(const std::size_t, const std::size_t, const float* const);
template void test::qr<half, half, float, true>(const std::size_t, const std::size_t, const float* const);
template void test::qr<half, half, float, false>(const std::size_t, const std::size_t, const float* const);
template void test::qr<float, float, float, false>(const std::size_t, const std::size_t, const float* const);
template void test::qr<float, float, float, true>(const std::size_t, const std::size_t, const float* const);

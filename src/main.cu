#include <iostream>
#include <random>
#include <cutf/device.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "tcqr.hpp"
#include "utils.hpp"

// #define PRINT_MATRIX

constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr float rand_range = 1.0f;
constexpr bool use_tc = true;
using input_t = half;
using output_t = half;
using norm_t = half;

namespace{
template <class T>std::string get_type_name();
template <> std::string get_type_name<float>(){return "float";};
template <> std::string get_type_name<half>(){return "half";};
}

int main(int argc, char** argv){
	// print device information {{{
	const auto device_props = cutf::cuda::device::get_properties_vector();
	for(auto device_id = 0; device_id < device_props.size(); device_id++){
		const auto &prop = device_props[device_id];
		utils::print_value(device_id, "Device id");
		utils::print_value(std::to_string(prop.major) + "." + std::to_string(prop.minor), "Compute capability");
		utils::print_value(prop.name, "Device name");
		utils::print_value(prop.totalGlobalMem/(1<<20), "Global memory", "MB");
		std::cout<<std::endl;
	}
	// }}}

	auto d_matrix_a = cutf::cuda::memory::get_device_unique_ptr<input_t>(M * N);
	auto d_matrix_r = cutf::cuda::memory::get_device_unique_ptr<output_t>(M * N);
	auto d_matrix_q = cutf::cuda::memory::get_device_unique_ptr<output_t>(M * M);
	auto d_matrix_qr = cutf::cuda::memory::get_device_unique_ptr<output_t>(M * N);

	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<input_t>(M * N);
	auto h_matrix_r = cutf::cuda::memory::get_host_unique_ptr<input_t>(M * N);
	auto h_matrix_q = cutf::cuda::memory::get_host_unique_ptr<input_t>(M * M);
	auto h_matrix_qr = cutf::cuda::memory::get_host_unique_ptr<input_t>(M * N);

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-rand_range, rand_range);
	for(std::size_t i = 0; i < M * N; i++){
		h_matrix_a.get()[i] = cutf::cuda::type::cast<input_t>(dist(mt));
	}

	// print type information{{{
	utils::print_value(get_type_name<input_t>(), "Input type");
	utils::print_value(get_type_name<output_t>(), "Output type");
	utils::print_value(get_type_name<norm_t>(), "Norm type");
	utils::print_value((use_tc ? "true" : "false"), "Use TC?");
	// }}}

	cutf::cuda::memory::copy(d_matrix_a.get(), h_matrix_a.get(), M * N);
	auto elapsed_time = utils::get_elapsed_time(
			[&d_matrix_q, &d_matrix_r, &d_matrix_a](){
			tcqr::qr16x16<input_t, output_t, norm_t, use_tc>(d_matrix_q.get(), d_matrix_r.get(), d_matrix_a.get(), M, N);
			cudaDeviceSynchronize();
			});
	cutf::cuda::memory::copy(h_matrix_q.get(), d_matrix_q.get(), M * M);
	cutf::cuda::memory::copy(h_matrix_r.get(), d_matrix_r.get(), M * N);
	utils::print_value(elapsed_time, "Elapsed time", "ms");

	// 検証
	output_t one = cutf::cuda::type::cast<output_t>(1.0f);
	output_t zero = cutf::cuda::type::cast<output_t>(0.0f);
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			M, N, M,
			&one,
			d_matrix_q.get(), M,
			d_matrix_r.get(), M,
			&zero,
			d_matrix_qr.get(), M
			);
	cutf::cuda::memory::copy(h_matrix_qr.get(), d_matrix_qr.get(), M * N);

#ifdef PRINT_MATRIX
	utils::print_matrix(h_matrix_a.get(), M, N, std::string("A").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_q.get(), M, M, std::string("Q").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_r.get(), M, N, std::string("R").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_qr.get(), M, N, std::string("QR").c_str());
#endif

	const auto error = utils::get_error(h_matrix_a.get(), h_matrix_qr.get(), M, N);
	utils::print_value(error , "error");
}

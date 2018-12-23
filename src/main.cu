#include <iostream>
#include <random>
#include <cutf/device.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "tcqr.hpp"
#include "utils.hpp"

constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr float rand_range = 1.0f;
constexpr bool use_tc = true;
using input_t = half;
using output_t = half;
using norm_t = half;

int main(int argc, char** argv){
	// print device information {{{
	const auto device_props = cutf::cuda::device::get_properties_vector();
	for(auto device_id = 0; device_id < device_props.size(); device_id++){
		const auto &prop = device_props[device_id];
		std::cout
			<<"# device "<<device_id<<std::endl
			<<"  - device name        : "<<prop.name<<std::endl
			<<"  - compute capability : "<<prop.major<<"."<<prop.minor<<std::endl
			<<"  - global memory      : "<<(prop.totalGlobalMem/(1<<20))<<" MB"<<std::endl;
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

	cutf::cuda::memory::copy(d_matrix_a.get(), h_matrix_a.get(), M * N);
	tcqr::qr16x16<input_t, output_t, norm_t, use_tc>(d_matrix_q.get(), d_matrix_r.get(), d_matrix_a.get(), M, N);
	cutf::cuda::memory::copy(h_matrix_q.get(), d_matrix_q.get(), M * M);
	cutf::cuda::memory::copy(h_matrix_r.get(), d_matrix_r.get(), M * N);

	utils::print_matrix(h_matrix_a.get(), M, N, std::string("A").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_q.get(), M, M, std::string("Q").c_str());
	std::cout<<std::endl;
	utils::print_matrix(h_matrix_r.get(), M, N, std::string("R").c_str());
	std::cout<<std::endl;


	// 検証
	output_t one = cutf::cuda::type::cast<output_t>(1.0f);
	output_t zero = cutf::cuda::type::cast<output_t>(0.0f);
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, M,
			&one,
			d_matrix_r.get(), M,
			d_matrix_q.get(), M,
			&zero,
			d_matrix_qr.get(), M
			);
	cutf::cuda::memory::copy(h_matrix_qr.get(), d_matrix_qr.get(), M * N);
	utils::print_matrix(h_matrix_qr.get(), M, N, std::string("R").c_str());
}

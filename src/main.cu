#include <iostream>
#include <random>
#include <cutf/device.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "utils.hpp"
#include "test.hpp"
#include "eigenqr.hpp"

constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr float rand_range = 1.0f;

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

	auto h_matrix_a = cutf::cuda::memory::get_host_unique_ptr<float>(M * N);
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-rand_range, rand_range);
	for(std::size_t i = 0; i < M * N; i++){
		h_matrix_a.get()[i] = dist(mt) * (i % (M + 1) == 0 ? 30.0f : 1.0f);
	}
	
	test::qr<float, float, float, true>(M, N, h_matrix_a.get());
	test::qr<float, float, float, false>(M, N, h_matrix_a.get());
	test::qr<half, half, half, true>(M, N, h_matrix_a.get());
	test::qr<half, half, half, false>(M, N, h_matrix_a.get());
	test::qr<half, half, float, true>(M, N, h_matrix_a.get());
	test::qr<half, half, float, false>(M, N, h_matrix_a.get());

	test::eigen<float, float, false>(M, h_matrix_a.get());
	test::eigen<float, float, true>(M, h_matrix_a.get());
	test::eigen<half, half, false>(M, h_matrix_a.get());
	test::eigen<half, half, true>(M, h_matrix_a.get());
	test::eigen<half, float, false>(M, h_matrix_a.get());
	test::eigen<half, float, true>(M, h_matrix_a.get());

	eigenqr::eigen16x16(nullptr, h_matrix_a.get(), M);
}

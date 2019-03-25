#include <iostream>
#include <random>
#include <cutf/device.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "utils.hpp"
#include "test.hpp"
#include "eigenqr.hpp"

#define QR_PRECISION_TEST
#define QR_TIME_TEST
#define BATCHED_QR_TIME_TEST
#define EIGEN_PRECISION_TEST
#define EIGEN_TIME_TEST


constexpr std::size_t batch_size = 1024;
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
		h_matrix_a.get()[i] = dist(mt);
	}

#ifdef QR_TIME_TEST
	std::cout<<"//---------------- qr time test"<<std::endl;
	test::time::qr<float, float, true>(M, N, h_matrix_a.get());
	test::time::qr<float, float, false>(M, N, h_matrix_a.get());
	test::time::qr<half, half, true>(M, N, h_matrix_a.get());
	test::time::qr<half, half, false>(M, N, h_matrix_a.get());
	test::time::qr<half, float, true>(M, N, h_matrix_a.get());
	test::time::qr<half, float, false>(M, N, h_matrix_a.get());
#endif

#ifdef QR_PRECISION_TEST
	std::cout<<"//---------------- qr precision test"<<std::endl;
	test::precision::qr<float, float, true>(M, N);
	test::precision::qr<float, float, false>(M, N);
	test::precision::qr<half, half, true>(M, N);
	test::precision::qr<half, half, false>(M, N);
	test::precision::qr<half, float, true>(M, N);
	test::precision::qr<half, float, false>(M, N);
#endif

#ifdef EIGEN_TIME_TEST
	// 対称行列の固有値計算
	for(std::size_t m = 0; m < M; m++){
		for(std::size_t n = m; n < N; n++){
			const auto val = dist(mt);
			h_matrix_a.get()[m + M * n] = val;
			h_matrix_a.get()[n + N * m] = val;
		}
	}

	std::cout<<"//---------------- eigenvalue time test"<<std::endl;
	test::time::eigen<float, float, false>(M, h_matrix_a.get());
	test::time::eigen<float, float, true>(M, h_matrix_a.get());
	test::time::eigen<half, half, false>(M, h_matrix_a.get());
	test::time::eigen<half, half, true>(M, h_matrix_a.get());
	test::time::eigen<half, float, false>(M, h_matrix_a.get());
	test::time::eigen<half, float, true>(M, h_matrix_a.get());
#endif

#ifdef EIGEN_PRECISION_TEST
	std::cout<<"//---------------- eigenvalue precision test"<<std::endl;
	test::precision::eigen_all(M);
#endif

#ifdef BATCHED_QR_TIME_TEST
	std::cout<<"//---------------- qr time test (batched)"<<std::endl;
	test::time::qr_batched<float, float, true>(M, N, batch_size);
	test::time::qr_batched<float, float, false>(M, N, batch_size);
	test::time::qr_batched<half, half, true>(M, N, batch_size);
	test::time::qr_batched<half, half, false>(M, N, batch_size);
	test::time::qr_batched<half, float, true>(M, N, batch_size);
	test::time::qr_batched<half, float, false>(M, N, batch_size);
#endif
}

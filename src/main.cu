#include <iostream>
#include <cutf/device.hpp>
#include "tcqr.hpp"

constexpr std::size_t M = 16;
constexpr std::size_t N = 16;

int main(int argc, char** argv){
	// print device information {{{
	const auto device_props = cutf::cuda::device::get_properties_vector();
	for(auto device_id = 0; device_id < device_props.size(); device_id++){
		const auto &prop = device_props[device_id];
		std::cout
			<<"# device "<<device_id<<std::endl
			<<"  - device name        : "<<prop.name<<std::endl
			<<"  - compute capability : "<<prop.major<<"."<<prop.minor<<std::endl;
	}
	// }}}
}

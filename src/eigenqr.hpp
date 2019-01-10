#ifndef __EIGENQR_HPP__
#define __EIGENQR_HPP__
namespace eigenqr{
void eigen16x16(float* const eigenvalues, const float* const a, const std::size_t n);
bool is_real(const float* const a, const std::size_t n);
}

#endif /* end of include guard */

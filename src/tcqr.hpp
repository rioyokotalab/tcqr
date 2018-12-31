#ifndef __TCQR_HPP__
#define __TCQR_HPP__
#include <cutf/cublas.hpp>

namespace tcqr{
template <class Input_t, class Output_t, class Norm_t, bool UseTC>
void qr16x16(Output_t* const q, Output_t* const r, const Input_t* const a, const std::size_t m, const std::size_t n);
template <class T, bool QR_UseTC, bool Eigen_UseTC, std::size_t L>
void eigen16x16(cublasHandle_t cublas, const T* eigen, const T* const a, std::size_t n);
}

#endif /* end of include guard */

#ifndef __TCQR_HPP__
#define __TCQR_HPP__

namespace tcqr{
template <class T, class Norm_t, bool UseTC>
void qr16x16(T* const q, T* const r, const T* const a, const std::size_t m, const std::size_t n);
template <class T, class Norm_t, bool UseTC>
void eigen16x16(T* const eigenvalues, const T* const a, std::size_t n);
}

#endif /* end of include guard */

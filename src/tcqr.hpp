#ifndef __TCQR_HPP__
#define __TCQR_HPP__

namespace tcqr{
template <class Input_t, class Output_t, class Norm_t, bool UseTC>
void qr16x16(Output_t* const q, Output_t* const r, const Input_t* const a, const std::size_t m, const std::size_t n);
}

#endif /* end of include guard */

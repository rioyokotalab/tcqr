#ifndef __TEST_HPP__
#define __TEST_HPP__

namespace test{
// 入力 a (m x n)を与えてQR分解を行い，精度と計算時間を表示する
template <class Input_t, class Output_t, class Norm_t, bool Use_TC>
void qr(const std::size_t m, const std::size_t n, const float* const a);
}

#endif /* end of include guard */

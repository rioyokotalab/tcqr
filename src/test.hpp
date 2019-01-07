#ifndef __TEST_HPP__
#define __TEST_HPP__

namespace test{
namespace time{
// 入力 a (m x n)を与えてQR分解を行い，精度と計算時間を表示する
template <class Input_t, class Output_t, class Norm_t, bool UseTC, std::size_t test_count = 4096>
void qr(const std::size_t m, const std::size_t n, const float* const a);
template <class T, class Norm_t, bool UseTC, std::size_t test_count = 1>
void eigen(const std::size_t n, const float* const a);
}
namespace precision{
// 入力 a (m x n)を与えてQR分解を行い，精度と計算時間を表示する
template <class Input_t, class Output_t, class Norm_t, bool UseTC, std::size_t test_count = 4096>
void qr(const std::size_t m, const std::size_t n);
template <class T, class Norm_t, bool UseTC, std::size_t test_count = 1>
void eigen(const std::size_t n, const float* const a);
}
}

#endif /* end of include guard */

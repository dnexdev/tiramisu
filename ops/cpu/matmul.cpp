#include "tiramisu/ops/matmul.hpp"
#include <stdexcept>

namespace tiramisu {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("Matmul currently only supports Float32.");
  }
  if (a.shape().size() != 2 || b.shape().size() != 2) {
    throw std::invalid_argument("Naive Matmul only supports 2D tensors.");
  }

  int64_t M = a.shape()[0];
  int64_t K = a.shape()[1];
  int64_t N = b.shape()[1];
  if (K != b.shape()[0]) {
    throw std::invalid_argument("Inner dimensions must match for matmul.");
  }

  Tensor c_a = a.contiguous();
  Tensor c_b = b.contiguous();
  Tensor out({M, N}, a.dtype(), a.device());

  const float* data_a = c_a.data<float>();
  const float* data_b = c_b.data<float>();
  float* data_out = out.data<float>();

  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += data_a[i * K + k] * data_b[k * N + j];
      }
      data_out[i * N + j] = sum;
    }
  }
  return out;
}

}
}
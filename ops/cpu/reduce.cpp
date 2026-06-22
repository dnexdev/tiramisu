#include "tiramisu/ops/reduce.hpp"

#include <stdexcept>

namespace tiramisu {
namespace ops {

Tensor sum(const Tensor& t) {
  if (t.dtype() != DType::Float32)
    throw std::runtime_error("Only Float32 supported.");

  Tensor c_t = t.contiguous();
  const float* data = c_t.data<float>();

  float total = 0.0f;
  int64_t elements = c_t.numel();
  for (int64_t i = 0; i < elements; i++) {
    total += data[i];
  }

  Tensor out({1}, t.dtype(), t.device());
  out.data<float>()[0] = total;
  return out;
}

Tensor mean(const Tensor& t) {
  Tensor s = sum(t);
  s.data<float>()[0] /= static_cast<float>(t.numel());
  return s;
}

}  // namespace ops
}  // namespace tiramisu
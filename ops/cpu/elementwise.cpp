#include "tiramisu/ops/elementwise.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "tiramisu/ops/broadcast.hpp"

namespace tiramisu {
namespace ops {

namespace {

// internal helper: left-pads shapes and strides to a target rank
// if a dimension is 1 (meaning it will be broadcasted), its stride becomes 0
void pad_to_rank(const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& strides, int64_t target_rank,
                 std::vector<int64_t>& out_shape,
                 std::vector<int64_t>& out_strides) {
  out_shape.resize(target_rank, 1);
  out_strides.resize(target_rank, 0);

  int64_t diff = target_rank - shape.size();
  for (size_t i = 0; i < shape.size(); i++) {
    out_shape[diff + i] = shape[i];
    out_strides[diff + i] = (shape[i] == 1) ? 0 : strides[i];
  }
}
}  // namespace

// Universal N-dimensional Broadcasting Binary Engine
template <typename Func>
Tensor apply_binary_op(const Tensor& a, const Tensor& b, Func op) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("Binary ops currently only support Float32.");
  }

  std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
  Tensor out(out_shape, a.dtype(), a.device());
  int64_t rank = out_shape.size();

  std::vector<int64_t> a_shape, a_strides, b_shape, b_strides;

  pad_to_rank(a.shape(), a.strides(), rank, a_shape, a_strides);
  pad_to_rank(b.shape(), b.strides(), rank, b_shape, b_strides);

  const float* a_data = a.data<float>();
  const float* b_data = b.data<float>();
  float* out_data = out.data<float>();

  int64_t total_elements = out.numel();

  for (int64_t i = 0; i < total_elements; i++) {
    int64_t temp = i;
    int64_t flat_idx_a = 0;
    int64_t flat_idx_b = 0;

    for (int d = rank - 1; d >= 0; d--) {
      int64_t coord = temp % out_shape[d];
      flat_idx_a += (coord % a_shape[d]) * a_strides[d];
      flat_idx_b += (coord % b_shape[d]) * b_strides[d];
      temp /= out_shape[d];
    }

    out_data[i] = op(a_data[flat_idx_a], b_data[flat_idx_b]);
  }

  return out;
}

// Universal Contiguous Unary Engine
template <typename Func>
Tensor apply_unary_op(const Tensor& t, Func op) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("Unary ops currently only support Float32");
  }

  Tensor c_t = t.contiguous();
  Tensor out(c_t.shape(), c_t.dtype(), c_t.device());

  const float* in_data = c_t.data<float>();
  float* out_data = out.data<float>();
  int64_t total = out.numel();

  for (int64_t i = 0; i < total; i++) {
    out_data[i] = op(in_data[i]);
  }

  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  return tiramisu::ops::apply_binary_op(a, b,
                                        [](float x, float y) { return x + y; });
}
Tensor sub(const Tensor& a, const Tensor& b) {
  return tiramisu::ops::apply_binary_op(a, b,
                                        [](float x, float y) { return x - y; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
  return tiramisu::ops::apply_binary_op(a, b,
                                        [](float x, float y) { return x * y; });
}
Tensor div(const Tensor& a, const Tensor& b) {
  return tiramisu::ops::apply_binary_op(a, b,
                                        [](float x, float y) { return x / y; });
}

Tensor neg(const Tensor& t) {
  return tiramisu::ops::apply_unary_op(t, [](float x) { return -x; });
}
Tensor exp(const Tensor& t) {
  return tiramisu::ops::apply_unary_op(t, [](float x) { return std::exp(x); });
}
Tensor log(const Tensor& t) {
  return tiramisu::ops::apply_unary_op(t, [](float x) { return std::log(x); });
}
Tensor relu(const Tensor& t) {
  return tiramisu::ops::apply_unary_op(
      t, [](float x) { return std::max(0.0f, x); });
}

namespace {

constexpr float kGeluSqrt2OverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;

float gelu_forward(float x) {
  float x3 = x * x * x;
  float inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x3);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

}  // namespace

Tensor gelu(const Tensor& t) {
  return tiramisu::ops::apply_unary_op(t, gelu_forward);
}

}  // namespace ops
}  // namespace tiramisu
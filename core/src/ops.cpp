#include "tiramisu/core/ops.hpp"
#include "tiramisu/core/tensor.hpp"
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace tiramisu {

namespace {

// internal helper: left-pads shapes and strides to a target rank
// if a dimension is 1 (meaning it will be broadcasted), its stride becomes 0
void pad_to_rank(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                 int64_t target_rank,
                std::vector<int64_t>& out_shape, std::vector<int64_t>& out_strides) {
  out_shape.resize(target_rank, 1);
  out_strides.resize(target_rank, 0);
  
  int64_t diff = target_rank - shape.size();
  for (size_t i = 0; i < shape.size(); i++) {
    out_shape[diff + i] = shape[i];
    out_strides[diff + i] = (shape[i] == 1) ? 0 : strides[i];
  }
  }
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, 
                                      const std::vector<int64_t>& shape_b) {
  int64_t rank_a = shape_a.size();
  int64_t rank_b = shape_b.size();
  int64_t out_rank = std::max(rank_a, rank_b);

  std::vector<int64_t> out_shape(out_rank);
  
  for (int64_t i = 0; i < out_rank; i++) {
    int64_t dim_a = (i < out_rank - rank_a) ? 1 : shape_a[i - (out_rank - rank_a)];
    int64_t dim_b = (i < out_rank - rank_b) ? 1 : shape_b[i - (out_rank - rank_b)];

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::invalid_argument("Tensor shapes are not broadcastable.");
    }
    out_shape[i] = std::max(dim_a, dim_b);
  }
  return out_shape;
}

Tensor add(const Tensor& a, const Tensor& b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("ops::add currently only supports Float32 scaffold.");
  }

  std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
  Tensor out(out_shape, a.dtype(), a.device());
  int64_t rank = out_shape.size();

  std::vector<int64_t> a_shape, a_strides, b_shape, b_strides;
  pad_to_rank(a.shape(), a.strides(), rank, a_shape, a_strides);
  pad_to_rank(b.shape(), b.strides(), rank, b_shape, b_strides);

  const float* a_data = a.data<float>();
  const float* b_data = b.data<float>();
  float *out_data = out.data<float>();
  
  int64_t total_elements = out.numel();
  // const auto& out_strides = out.strides();

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

    out_data[i] = a_data[flat_idx_a] + b_data[flat_idx_b];
  }

  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("ops::mul currently only supports Float32 scaffold.");
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

    out_data[i] = a_data[flat_idx_a] * b_data[flat_idx_b];
  }

  return out;
}

Tensor sum(const Tensor& t) {
  if (t.dtype() != DType::Float32) throw std::runtime_error("Only Float32 supported.");

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


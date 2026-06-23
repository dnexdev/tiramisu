#include "tiramisu/ops/normalization.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tiramisu::ops {

Tensor softmax(const Tensor& x) {
  if (x.dtype() != DType::Float32) {
    throw std::runtime_error("softmax: only Float32 supported");
  }

  auto shape = x.shape();
  int64_t N = shape.back();
  int64_t rows = x.numel() / N;

  Tensor c_x = x.contiguous();
  Tensor out(shape);

  const float* src = c_x.data<float>();
  float* dst = out.data<float>();

  for (int64_t r = 0; r < rows; r++) {
    const float* row_in = src + r * N;
    float* row_out = dst + r * N;

    float max_val = row_in[0];
    for (int64_t i = 1; i < N; i++) {
      if (row_in[i] > max_val) {
        max_val = row_in[i];
      }
    }
    
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < N; i++) {
      row_out[i] = std::exp(row_in[i] - max_val);
      sum_exp += row_out[i];
    }

    for (int64_t i = 0; i < N; i++) {
      row_out[i] /= sum_exp;
    }
  }
  return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
  if (x.shape().size() < 2) {
    throw std::invalid_argument("layernorm: input must be at least 2D");
  }

  int64_t N = x.shape().back();
  int64_t rows = x.numel() / N;

  Tensor c_x = x.contiguous();
  Tensor out(x.shape());

  const float* xd = c_x.data<float>();
  const float* gd = gamma.data<float>();
  const float* bd = beta.data<float>();
  float* od = out.data<float>();

  for (int64_t r = 0; r < rows; r++) {
    const float* row = xd + r * N;
    float* o = od + r * N;

    float mean = 0.0f;
    for (int64_t i = 0; i < N; i++) {
      mean += row[i];
    }
    mean /= static_cast<float>(N);

    float var = 0.0f;
    for (int64_t i = 0; i < N; i++) {
      float diff = row[i] - mean;
      var += diff * diff;
    }
    var /= N;

    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int64_t i = 0; i < N; i++) {
      float x_hat = (row[i] - mean) * inv_std;
      o[i] = gd[i] * x_hat + bd[i];
    }
  }
  return out;

}

}
#include "tiramisu/ops/normalization.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "tiramisu/core/device.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include "tiramisu/ops/cuda_ops.hpp"
#endif

namespace tiramisu::ops {

Tensor softmax(const Tensor& x) {
#ifdef TIRAMISU_CUDA_ENABLED
  if (x.device() == Device::CUDA) {
    return cuda::softmax(x);
  }
#endif
  if (x.dtype() != DType::Float32) {
    throw std::runtime_error("softmax: only Float32 supported");
  }
  // Softmax is applied over the last dimension; a rank-0 tensor has no such
  // dimension and would divide by zero when computing `rows`. Rank-1 input
  // is valid (a single row of length N).
  if (x.shape().empty()) {
    throw std::invalid_argument("softmax: input must be at least 1D");
  }

  auto shape = x.shape();
  int64_t N = shape.back();
  int64_t rows = x.numel() / N;

  Tensor c_x = x.contiguous();
  Tensor out(shape);

  const float* src = c_x.data<float>();
  float* dst = out.data<float>();

  // Numerically stable softmax: subtract the per-row max before exp so the
  // largest exponent argument is 0. Without this, exp(row_in[i]) overflows
  // for inputs >~ 88 (fp32 exp is finite up to log(FLT_MAX) ≈ 88.7).
  // Standard log-sum-exp trick — see Goodfellow, Bengio, Courville
  // "Deep Learning" §4.1. `sum_exp` is guaranteed ≥ 1 (the max-argument
  // term evaluates to exp(0)=1), so the divide is always safe.
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

// Layer Normalization (Ba, Kiros, Hinton 2016):
//   https://arxiv.org/abs/1607.06450
// For each row of shape [..., N]:
//   µ = mean(row),   σ² = var(row)
//   ŷ = (x − µ) / √(σ² + eps)
//   out = γ · ŷ + β
// Statistics are computed per-row over the last dimension; γ, β are learned
// per-feature (shape {N}).
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
#ifdef TIRAMISU_CUDA_ENABLED
  if (x.device() == Device::CUDA) {
    return cuda::layernorm(x, gamma, beta, eps);
  }
#endif
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
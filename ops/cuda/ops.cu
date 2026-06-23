#include "tiramisu/ops/cuda_ops.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "tiramisu/core/device.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/broadcast.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include <cuda_runtime.h>

namespace tiramisu::ops::cuda {

namespace {

__global__ void matmul_2d_kernel(const float* a, const float* b, float* c,
                                 int64_t M, int64_t K, int64_t N) {
  int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float sum = 0.0f;
  for (int64_t k = 0; k < K; ++k) {
    sum += a[row * K + k] * b[k * N + col];
  }
  c[row * N + col] = sum;
}

__global__ void binary_kernel(const float* a, const float* b, float* out,
                              int64_t n, int op) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (op == 0) {
    out[i] = a[i] + b[i];
  } else {
    out[i] = a[i] * b[i];
  }
}

__global__ void gelu_kernel(const float* in, float* out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = in[i];
  out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void relu_kernel(const float* in, float* out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void softmax_rows_kernel(const float* in, float* out, int64_t rows,
                                    int64_t cols) {
  int64_t r = blockIdx.x;
  if (r >= rows) return;
  const float* row_in = in + r * cols;
  float* row_out = out + r * cols;
  float max_val = row_in[0];
  for (int64_t c = 1; c < cols; ++c) {
    if (row_in[c] > max_val) max_val = row_in[c];
  }
  float sum = 0.0f;
  for (int64_t c = 0; c < cols; ++c) {
    row_out[c] = expf(row_in[c] - max_val);
    sum += row_out[c];
  }
  for (int64_t c = 0; c < cols; ++c) {
    row_out[c] /= sum;
  }
}

__global__ void layernorm_kernel(const float* x, const float* gamma,
                                 const float* beta, float* out, int64_t rows,
                                 int64_t cols, float eps) {
  int64_t r = blockIdx.x;
  if (r >= rows) return;
  const float* row = x + r * cols;
  float* o = out + r * cols;
  float mean = 0.0f;
  for (int64_t i = 0; i < cols; ++i) mean += row[i];
  mean /= static_cast<float>(cols);
  float var = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    float d = row[i] - mean;
    var += d * d;
  }
  var /= static_cast<float>(cols);
  float inv_std = rsqrtf(var + eps);
  for (int64_t i = 0; i < cols; ++i) {
    o[i] = gamma[i] * (row[i] - mean) * inv_std + beta[i];
  }
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
  const auto& shape_a = a.shape();
  const auto& shape_b = b.shape();
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::runtime_error("cuda matmul: only 2D tensors supported");
  }

  const int64_t M = shape_a[0];
  const int64_t K = shape_a[1];
  const int64_t N = shape_b[1];

  Tensor c_a = a.contiguous();
  Tensor c_b = b.contiguous();
  Tensor out({M, N}, DType::Float32, Device::CUDA);

  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(N) + block.x - 1) / block.x,
            (static_cast<unsigned>(M) + block.y - 1) / block.y);
  matmul_2d_kernel<<<grid, block>>>(c_a.data<float>(), c_b.data<float>(),
                                      out.data<float>(), M, K, N);
  cudaDeviceSynchronize();
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor c_a = a.contiguous();
  Tensor c_b = b.contiguous();
  if (c_a.numel() != c_b.numel()) {
    throw std::runtime_error("cuda add: requires same numel after broadcast");
  }
  Tensor out(c_a.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((out.numel() + block - 1) / block);
  binary_kernel<<<grid, block>>>(c_a.data<float>(), c_b.data<float>(),
                                   out.data<float>(), out.numel(), 0);
  cudaDeviceSynchronize();
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor c_a = a.contiguous();
  Tensor c_b = b.contiguous();
  if (c_a.numel() != c_b.numel()) {
    throw std::runtime_error("cuda mul: requires same numel after broadcast");
  }
  Tensor out(c_a.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((out.numel() + block - 1) / block);
  binary_kernel<<<grid, block>>>(c_a.data<float>(), c_b.data<float>(),
                                   out.data<float>(), out.numel(), 1);
  cudaDeviceSynchronize();
  return out;
}

Tensor gelu(const Tensor& t) {
  Tensor c = t.contiguous();
  Tensor out(t.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((t.numel() + block - 1) / block);
  gelu_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), t.numel());
  cudaDeviceSynchronize();
  return out;
}

Tensor relu(const Tensor& t) {
  Tensor c = t.contiguous();
  Tensor out(t.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((t.numel() + block - 1) / block);
  relu_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), t.numel());
  cudaDeviceSynchronize();
  return out;
}

Tensor softmax(const Tensor& x) {
  Tensor c = x.contiguous();
  Tensor out(x.shape(), DType::Float32, Device::CUDA);
  const int64_t cols = x.shape().back();
  const int64_t rows = x.numel() / cols;
  softmax_rows_kernel<<<static_cast<int>(rows), 1>>>(
      c.data<float>(), out.data<float>(), rows, cols);
  cudaDeviceSynchronize();
  return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps) {
  Tensor c = x.contiguous();
  Tensor out(x.shape(), DType::Float32, Device::CUDA);
  const int64_t cols = x.shape().back();
  const int64_t rows = x.numel() / cols;
  layernorm_kernel<<<static_cast<int>(rows), 1>>>(
      c.data<float>(), gamma.data<float>(), beta.data<float>(),
      out.data<float>(), rows, cols, eps);
  cudaDeviceSynchronize();
  return out;
}

}  // namespace tiramisu::ops::cuda

#endif  // TIRAMISU_CUDA_ENABLED

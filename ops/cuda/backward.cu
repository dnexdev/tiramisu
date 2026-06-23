#include "tiramisu/ops/cuda_ops.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "tiramisu/core/cuda_common.hpp"
#include "tiramisu/core/tensor.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include <cuda_runtime.h>

namespace tiramisu::ops::cuda {

namespace {

__global__ void reduce_sum_rows_kernel(const float* src, float* dst, int64_t rows,
                                       int64_t cols) {
  const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) {
    return;
  }
  float sum = 0.0f;
  for (int64_t r = 0; r < rows; ++r) {
    sum += src[r * cols + c];
  }
  dst[c] = sum;
}

__global__ void gelu_backward_kernel(const float* grad, const float* x, float* gx,
                                     int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float xi = x[i];
  const float x2 = xi * xi;
  const float x3 = x2 * xi;
  const float inner = 0.7978845608f * (xi + 0.044715f * x3);
  const float tanh_inner = tanhf(inner);
  const float sech2 = 1.0f - tanh_inner * tanh_inner;
  const float inner_deriv = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x2);
  const float d_out_dx =
      0.5f * (1.0f + tanh_inner) + 0.5f * xi * sech2 * inner_deriv;
  gx[i] = grad[i] * d_out_dx;
}

__global__ void relu_backward_kernel(const float* grad, const float* x, float* gx,
                                     int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  gx[i] = x[i] > 0.0f ? grad[i] : 0.0f;
}

__global__ void softmax_backward_kernel(const float* grad_out, const float* out,
                                        float* grad_x, int64_t rows,
                                        int64_t cols) {
  const int64_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const float* go = grad_out + r * cols;
  const float* o = out + r * cols;
  float* gx = grad_x + r * cols;
  float dot = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    dot += go[i] * o[i];
  }
  for (int64_t i = 0; i < cols; ++i) {
    gx[i] = o[i] * (go[i] - dot);
  }
}

__global__ void layernorm_backward_kernel(
    const float* grad_out, const float* x, const float* gamma, float* grad_x,
    float* grad_gamma, float* grad_beta, int64_t rows, int64_t cols, float eps) {
  const int64_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const float* go = grad_out + r * cols;
  const float* xrow = x + r * cols;
  float* gx = grad_x + r * cols;

  float mean = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    mean += xrow[i];
  }
  mean /= static_cast<float>(cols);

  float var = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    const float d = xrow[i] - mean;
    var += d * d;
  }
  var /= static_cast<float>(cols);
  const float inv_std = rsqrtf(var + eps);

  float grad_var = 0.0f;
  float grad_mean = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    const float x_hat = (xrow[i] - mean) * inv_std;
    const float grad_x_hat = go[i] * gamma[i];
    atomicAdd(grad_gamma + i, go[i] * x_hat);
    atomicAdd(grad_beta + i, go[i]);
    grad_var += grad_x_hat * (xrow[i] - mean) * -0.5f * inv_std * inv_std * inv_std;
    grad_mean += grad_x_hat * -inv_std;
  }
  for (int64_t i = 0; i < cols; ++i) {
    const float grad_x_hat = go[i] * gamma[i];
    gx[i] = grad_x_hat * inv_std +
            grad_var * 2.0f * (xrow[i] - mean) / static_cast<float>(cols) +
            grad_mean / static_cast<float>(cols);
  }
}

__global__ void embedding_backward_kernel(const float* grad_out,
                                            const float* indices, float* grad_w,
                                            int64_t batch, int64_t seq,
                                            int64_t dim) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * seq;
  if (idx >= total) {
    return;
  }
  const int64_t token = static_cast<int64_t>(indices[idx]);
  const float* g = grad_out + idx * dim;
  float* gw = grad_w + token * dim;
  for (int64_t d = 0; d < dim; ++d) {
    atomicAdd(gw + d, g[d]);
  }
}

__global__ void merge_heads_backward_kernel(const float* grad_out, float* grad_in,
                                            int64_t batch, int64_t heads,
                                            int64_t seq, int64_t d_k,
                                            int64_t d_model) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * seq * d_model;
  if (idx >= total) {
    return;
  }
  const int64_t b = idx / (seq * d_model);
  const int64_t rem = idx % (seq * d_model);
  const int64_t s = rem / d_model;
  const int64_t d = rem % d_model;
  const int64_t h = d / d_k;
  const int64_t dk = d % d_k;
  const int64_t src_idx = (b * seq + s) * d_model + d;
  const int64_t dst_idx = ((b * heads + h) * seq + s) * d_k + dk;
  grad_in[dst_idx] = grad_out[src_idx];
}

__global__ void contiguous_backward_kernel(const float* grad_out, float* grad_in,
                                           int64_t numel, int64_t rank,
                                           const int64_t* shape,
                                           const int64_t* strides) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel) {
    return;
  }
  int64_t temp = i;
  int64_t dst_idx = 0;
  for (int64_t d = rank - 1; d >= 0; --d) {
    const int64_t coord = temp % shape[d];
    dst_idx += coord * strides[d];
    temp /= shape[d];
  }
  atomicAdd(grad_in + dst_idx, grad_out[i]);
}

__global__ void ce_forward_kernel(const float* logits, const float* targets,
                                  float* softmax_buf, float* loss_out,
                                  int64_t batch, int64_t C) {
  const int64_t i = blockIdx.x;
  if (i >= batch) {
    return;
  }
  const float* row = logits + i * C;
  float* sm = softmax_buf + i * C;
  float max_val = row[0];
  for (int64_t c = 1; c < C; ++c) {
    if (row[c] > max_val) {
      max_val = row[c];
    }
  }
  float sum = 0.0f;
  for (int64_t c = 0; c < C; ++c) {
    sm[c] = expf(row[c] - max_val);
    sum += sm[c];
  }
  for (int64_t c = 0; c < C; ++c) {
    sm[c] /= sum;
  }
  const int64_t target = static_cast<int64_t>(targets[i]);
  const float nll = -logf(sm[target] + 1e-9f);
  atomicAdd(loss_out, nll);
}

__global__ void ce_backward_kernel(const float upstream, const float* softmax,
                                   const float* targets, float* grad_logits,
                                   int64_t batch, int64_t C) {
  const int64_t i = blockIdx.x;
  if (i >= batch) {
    return;
  }
  const float* sm = softmax + i * C;
  float* gl = grad_logits + i * C;
  const int64_t target = static_cast<int64_t>(targets[i]);
  const float scale = upstream / static_cast<float>(batch);
  for (int64_t c = 0; c < C; ++c) {
    const float indicator = (c == target) ? 1.0f : 0.0f;
    gl[c] = scale * (sm[c] - indicator);
  }
}

__global__ void adamw_kernel(float* param, const float* grad, float* m, float* v,
                             int64_t n, float lr, float beta1, float beta2,
                             float eps, float weight_decay, float bias1,
                             float bias2) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float g = grad[i];
  m[i] = beta1 * m[i] + (1.0f - beta1) * g;
  v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
  const float m_hat = m[i] / bias1;
  const float v_hat = v[i] / bias2;
  param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
}

}  // namespace

Tensor reduce_sum_first_dim(const Tensor& t) {
  Tensor c = t.contiguous();
  const int64_t rows = c.shape()[0];
  const int64_t cols = c.numel() / rows;
  std::vector<int64_t> out_shape(c.shape().begin() + 1, c.shape().end());
  Tensor out(out_shape, DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((cols + block - 1) / block);
  reduce_sum_rows_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(),
                                          rows, cols);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor gelu_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor g = grad_output.contiguous();
  Tensor x = input.contiguous();
  Tensor grad_in(input.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((input.numel() + block - 1) / block);
  gelu_backward_kernel<<<grid, block>>>(g.data<float>(), x.data<float>(),
                                        grad_in.data<float>(), input.numel());
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_in;
}

Tensor relu_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor g = grad_output.contiguous();
  Tensor x = input.contiguous();
  Tensor grad_in(input.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((input.numel() + block - 1) / block);
  relu_backward_kernel<<<grid, block>>>(g.data<float>(), x.data<float>(),
                                        grad_in.data<float>(), input.numel());
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_in;
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor g = grad_output.contiguous();
  Tensor o = output.contiguous();
  Tensor grad_x(output.shape(), DType::Float32, Device::CUDA);
  const int64_t cols = output.shape().back();
  const int64_t rows = output.numel() / cols;
  softmax_backward_kernel<<<static_cast<int>(rows), 1>>>(
      g.data<float>(), o.data<float>(), grad_x.data<float>(), rows, cols);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_x;
}

Tensor layernorm_backward(const Tensor& grad_output, const Tensor& x,
                          const Tensor& gamma, float eps, Tensor& grad_gamma,
                          Tensor& grad_beta) {
  Tensor g = grad_output.contiguous();
  Tensor xc = x.contiguous();
  Tensor grad_x(x.shape(), DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(grad_gamma.data<float>(), 0,
                                 static_cast<size_t>(gamma.numel()) * sizeof(float)));
  TIRAMISU_CUDA_CHECK(cudaMemset(grad_beta.data<float>(), 0,
                                 static_cast<size_t>(gamma.numel()) * sizeof(float)));
  const int64_t cols = x.shape().back();
  const int64_t rows = x.numel() / cols;
  layernorm_backward_kernel<<<static_cast<int>(rows), 1>>>(
      g.data<float>(), xc.data<float>(), gamma.data<float>(),
      grad_x.data<float>(), grad_gamma.data<float>(), grad_beta.data<float>(),
      rows, cols, eps);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_x;
}

Tensor embedding_backward(const Tensor& grad_output, const Tensor& indices,
                          int64_t vocab, int64_t dim) {
  Tensor g = grad_output.contiguous();
  Tensor idx = indices.contiguous();
  const int64_t batch = indices.shape()[0];
  const int64_t seq = indices.shape()[1];
  Tensor grad_w({vocab, dim}, DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(grad_w.data<float>(), 0,
                                 static_cast<size_t>(vocab * dim) * sizeof(float)));
  const int block = 256;
  const int grid = static_cast<int>((batch * seq + block - 1) / block);
  embedding_backward_kernel<<<grid, block>>>(
      g.data<float>(), idx.data<float>(), grad_w.data<float>(), batch, seq, dim);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_w;
}

Tensor merge_heads_backward(const Tensor& grad_output, int64_t batch,
                            int64_t heads, int64_t seq, int64_t d_k,
                            int64_t d_model) {
  Tensor g = grad_output.contiguous();
  Tensor grad_in({batch, heads, seq, d_k}, DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(grad_in.data<float>(), 0,
                                 static_cast<size_t>(grad_in.numel()) * sizeof(float)));
  const int block = 256;
  const int64_t total = batch * seq * d_model;
  const int grid = static_cast<int>((total + block - 1) / block);
  merge_heads_backward_kernel<<<grid, block>>>(
      g.data<float>(), grad_in.data<float>(), batch, heads, seq, d_k, d_model);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_in;
}

Tensor contiguous_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor g = grad_output.contiguous();
  Tensor grad_in(input.shape(), DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(grad_in.data<float>(), 0,
                                 static_cast<size_t>(grad_in.numel()) * sizeof(float)));
  const int64_t rank = static_cast<int64_t>(input.shape().size());
  int64_t* d_shape = nullptr;
  int64_t* d_strides = nullptr;
  TIRAMISU_CUDA_CHECK(cudaMalloc(&d_shape, static_cast<size_t>(rank) * sizeof(int64_t)));
  TIRAMISU_CUDA_CHECK(cudaMalloc(&d_strides, static_cast<size_t>(rank) * sizeof(int64_t)));
  TIRAMISU_CUDA_CHECK(cudaMemcpy(d_shape, input.shape().data(),
                                 static_cast<size_t>(rank) * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
  TIRAMISU_CUDA_CHECK(cudaMemcpy(d_strides, input.strides().data(),
                                 static_cast<size_t>(rank) * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
  const int block = 256;
  const int grid = static_cast<int>((g.numel() + block - 1) / block);
  contiguous_backward_kernel<<<grid, block>>>(
      g.data<float>(), grad_in.data<float>(), g.numel(), rank, d_shape, d_strides);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  cudaFree(d_shape);
  cudaFree(d_strides);
  return grad_in;
}

Tensor cross_entropy_forward(const Tensor& logits, const Tensor& targets,
                             Tensor& softmax_out) {
  const int64_t batch = logits.shape()[0];
  const int64_t C = logits.shape()[1];
  Tensor c_logits = logits.contiguous();
  Tensor c_targets = targets.contiguous();
  Tensor loss({1}, DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(loss.data<float>(), 0, sizeof(float)));
  ce_forward_kernel<<<static_cast<int>(batch), 1>>>(
      c_logits.data<float>(), c_targets.data<float>(), softmax_out.data<float>(),
      loss.data<float>(), batch, C);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  float host_loss = 0.0f;
  TIRAMISU_CUDA_CHECK(cudaMemcpy(&host_loss, loss.data<float>(), sizeof(float),
                                 cudaMemcpyDeviceToHost));
  host_loss /= static_cast<float>(batch);
  TIRAMISU_CUDA_CHECK(cudaMemcpy(loss.data<float>(), &host_loss, sizeof(float),
                                 cudaMemcpyHostToDevice));
  return loss;
}

Tensor cross_entropy_backward(const Tensor& grad_scalar, const Tensor& softmax,
                              const Tensor& targets, int64_t batch, int64_t C) {
  float upstream = 0.0f;
  TIRAMISU_CUDA_CHECK(cudaMemcpy(&upstream, grad_scalar.data<float>(),
                                 sizeof(float), cudaMemcpyDeviceToHost));
  Tensor grad_logits({batch, C}, DType::Float32, Device::CUDA);
  ce_backward_kernel<<<static_cast<int>(batch), 1>>>(
      upstream, softmax.data<float>(), targets.data<float>(),
      grad_logits.data<float>(), batch, C);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return grad_logits;
}

void adamw_step(float* param, const float* grad, float* m, float* v, int64_t n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                int64_t t) {
  const float bias1 = 1.0f - powf(beta1, static_cast<float>(t));
  const float bias2 = 1.0f - powf(beta2, static_cast<float>(t));
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  adamw_kernel<<<grid, block>>>(param, grad, m, v, n, lr, beta1, beta2, eps,
                                weight_decay, bias1, bias2);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
}

void zero_grad(float* grad, int64_t n) {
  TIRAMISU_CUDA_CHECK(
      cudaMemset(grad, 0, static_cast<size_t>(n) * sizeof(float)));
}

}  // namespace tiramisu::ops::cuda

#endif  // TIRAMISU_CUDA_ENABLED

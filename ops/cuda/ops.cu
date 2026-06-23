#include "tiramisu/ops/cuda_ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tiramisu/core/cuda_common.hpp"
#include "tiramisu/core/device.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/broadcast.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include <cuda_runtime.h>

namespace tiramisu::ops::cuda {

namespace {

void pad_to_rank(const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& strides, int64_t target_rank,
                 std::vector<int64_t>& out_shape,
                 std::vector<int64_t>& out_strides) {
  out_shape.assign(target_rank, 1);
  out_strides.assign(target_rank, 0);
  const int64_t diff = target_rank - static_cast<int64_t>(shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    out_shape[diff + static_cast<int64_t>(i)] = shape[i];
    out_strides[diff + static_cast<int64_t>(i)] =
        (shape[i] == 1) ? 0 : strides[i];
  }
}

int64_t product(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

std::vector<int64_t> batch_dims(const std::vector<int64_t>& shape) {
  if (shape.size() < 2) {
    return {};
  }
  return std::vector<int64_t>(shape.begin(), shape.end() - 2);
}

std::vector<int64_t> pad_batch_shape(const std::vector<int64_t>& batch_shape,
                                     int64_t target_rank) {
  std::vector<int64_t> padded(target_rank, 1);
  for (size_t i = 0; i < batch_shape.size(); ++i) {
    padded[target_rank - batch_shape.size() + i] = batch_shape[i];
  }
  return padded;
}

std::vector<int64_t> batch_strides(const std::vector<int64_t>& shape) {
  const int64_t batch_rank = static_cast<int64_t>(shape.size()) - 2;
  if (batch_rank <= 0) {
    return {};
  }
  std::vector<int64_t> strides(batch_rank);
  int64_t stride = shape[batch_rank] * shape[batch_rank + 1];
  for (int64_t i = batch_rank - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

std::vector<int64_t> precompute_batch_offsets(
    int64_t num_batches, const std::vector<int64_t>& batch_out,
    const std::vector<int64_t>& padded_operand_batch,
    const std::vector<int64_t>& operand_batch_strides) {
  std::vector<int64_t> offsets(num_batches, 0);
  if (operand_batch_strides.empty() || num_batches == 0) {
    return offsets;
  }
  const int64_t batch_rank = static_cast<int64_t>(batch_out.size());
  std::vector<int64_t> coords(batch_rank, 0);
  for (int64_t batch = 0; batch < num_batches; ++batch) {
    int64_t offset = 0;
    for (int64_t i = 0; i < batch_rank; ++i) {
      const int64_t idx = padded_operand_batch[i] == 1 ? 0 : coords[i];
      offset += idx * operand_batch_strides[i];
    }
    offsets[batch] = offset;
    if (batch + 1 == num_batches) {
      break;
    }
    for (int64_t d = batch_rank - 1; d >= 0; --d) {
      coords[d]++;
      if (coords[d] == batch_out[d]) {
        coords[d] = 0;
        continue;
      }
      break;
    }
  }
  return offsets;
}

__global__ void matmul_2d_kernel(const float* a, const float* b, float* c,
                                 int64_t M, int64_t K, int64_t N) {
  const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }
  float sum = 0.0f;
  for (int64_t k = 0; k < K; ++k) {
    sum += a[row * K + k] * b[k * N + col];
  }
  c[row * N + col] = sum;
}

__global__ void broadcast_binary_kernel(const float* a, const float* b,
                                        float* out, int64_t numel, int64_t rank,
                                        const int64_t* out_shape,
                                        const int64_t* a_shape,
                                        const int64_t* a_strides,
                                        const int64_t* b_shape,
                                        const int64_t* b_strides, int op) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel) {
    return;
  }
  int64_t temp = i;
  int64_t flat_a = 0;
  int64_t flat_b = 0;
  for (int64_t d = rank - 1; d >= 0; --d) {
    const int64_t coord = temp % out_shape[d];
    flat_a += (coord % a_shape[d]) * a_strides[d];
    flat_b += (coord % b_shape[d]) * b_strides[d];
    temp /= out_shape[d];
  }
  const float av = a[flat_a];
  const float bv = b[flat_b];
  if (op == 0) {
    out[i] = av + bv;
  } else if (op == 1) {
    out[i] = av * bv;
  } else {
    out[i] = -av;
  }
}

__global__ void gelu_kernel(const float* in, float* out, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float x = in[i];
  out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void relu_kernel(const float* in, float* out, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void neg_kernel(const float* in, float* out, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = -in[i];
}

__global__ void sum_kernel(const float* in, float* out, int64_t n) {
  __shared__ float buf[256];
  const int tid = threadIdx.x;
  float local = 0.0f;
  for (int64_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += blockDim.x * gridDim.x) {
    local += in[i];
  }
  buf[tid] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buf[tid] += buf[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, buf[0]);
  }
}

__global__ void softmax_rows_kernel(const float* in, float* out, int64_t rows,
                                    int64_t cols) {
  const int64_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const float* row_in = in + r * cols;
  float* row_out = out + r * cols;
  float max_val = row_in[0];
  for (int64_t c = 1; c < cols; ++c) {
    if (row_in[c] > max_val) {
      max_val = row_in[c];
    }
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
  const int64_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const float* row = x + r * cols;
  float* o = out + r * cols;
  float mean = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    mean += row[i];
  }
  mean /= static_cast<float>(cols);
  float var = 0.0f;
  for (int64_t i = 0; i < cols; ++i) {
    const float d = row[i] - mean;
    var += d * d;
  }
  var /= static_cast<float>(cols);
  const float inv_std = rsqrtf(var + eps);
  for (int64_t i = 0; i < cols; ++i) {
    o[i] = gamma[i] * (row[i] - mean) * inv_std + beta[i];
  }
}

__global__ void embedding_kernel(const float* weight, const float* indices,
                                 float* out, int64_t batch, int64_t seq,
                                 int64_t dim) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * seq;
  if (idx >= total) {
    return;
  }
  const int64_t token = static_cast<int64_t>(indices[idx]);
  const float* row = weight + token * dim;
  float* dst = out + idx * dim;
  for (int64_t d = 0; d < dim; ++d) {
    dst[d] = row[d];
  }
}

__global__ void merge_heads_kernel(const float* src, float* dst, int64_t batch,
                                   int64_t heads, int64_t seq, int64_t d_k,
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
  const int64_t src_idx = ((b * heads + h) * seq + s) * d_k + dk;
  dst[idx] = src[src_idx];
}

void upload_i64(const std::vector<int64_t>& host, int64_t** device) {
  TIRAMISU_CUDA_CHECK(cudaMalloc(device, host.size() * sizeof(int64_t)));
  TIRAMISU_CUDA_CHECK(cudaMemcpy(*device, host.data(),
                                 host.size() * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
}

void free_i64(int64_t* device) { cudaFree(device); }

Tensor launch_broadcast_binary(const Tensor& a, const Tensor& b, int op) {
  const std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
  const int64_t rank = static_cast<int64_t>(out_shape.size());
  std::vector<int64_t> a_shape, a_strides, b_shape, b_strides;
  pad_to_rank(a.shape(), a.strides(), rank, a_shape, a_strides);
  pad_to_rank(b.shape(), b.strides(), rank, b_shape, b_strides);

  Tensor out(out_shape, DType::Float32, Device::CUDA);
  int64_t *d_out_shape = nullptr, *d_a_shape = nullptr, *d_a_strides = nullptr,
          *d_b_shape = nullptr, *d_b_strides = nullptr;
  upload_i64(out_shape, &d_out_shape);
  upload_i64(a_shape, &d_a_shape);
  upload_i64(a_strides, &d_a_strides);
  upload_i64(b_shape, &d_b_shape);
  upload_i64(b_strides, &d_b_strides);

  const int block = 256;
  const int grid = static_cast<int>((out.numel() + block - 1) / block);
  broadcast_binary_kernel<<<grid, block>>>(
      a.data<float>(), b.data<float>(), out.data<float>(), out.numel(), rank,
      d_out_shape, d_a_shape, d_a_strides, d_b_shape, d_b_strides, op);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  free_i64(d_out_shape);
  free_i64(d_a_shape);
  free_i64(d_a_strides);
  free_i64(d_b_shape);
  free_i64(d_b_strides);
  return out;
}

}  // namespace

void assert_same_device(const Tensor& a, const Tensor& b) {
  if (a.device() != b.device()) {
    throw std::runtime_error("cuda op: device mismatch");
  }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  assert_same_device(a, b);
  const auto& shape_a = a.shape();
  const auto& shape_b = b.shape();
  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::invalid_argument("cuda matmul: tensors must be at least 2D");
  }

  const int64_t M = shape_a[shape_a.size() - 2];
  const int64_t K = shape_a[shape_a.size() - 1];
  const int64_t K_b = shape_b[shape_b.size() - 2];
  const int64_t N = shape_b[shape_b.size() - 1];
  if (K != K_b) {
    throw std::invalid_argument("cuda matmul: inner dimensions must match");
  }

  const std::vector<int64_t> batch_a = batch_dims(shape_a);
  const std::vector<int64_t> batch_b = batch_dims(shape_b);
  const std::vector<int64_t> batch_out = broadcast_shapes(batch_a, batch_b);

  std::vector<int64_t> out_shape = batch_out;
  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor c_a = a.contiguous();
  Tensor c_b = b.contiguous();
  Tensor out(out_shape, DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(out.data<float>(), 0,
                                 static_cast<size_t>(out.numel()) * sizeof(float)));

  const int64_t num_batches = product(batch_out);
  const int64_t matrix_numel = M * N;
  const auto padded_a = pad_batch_shape(batch_a, batch_out.size());
  const auto padded_b = pad_batch_shape(batch_b, batch_out.size());
  const auto strides_a = batch_strides(shape_a);
  const auto strides_b = batch_strides(shape_b);
  const std::vector<int64_t> offsets_a =
      precompute_batch_offsets(num_batches, batch_out, padded_a, strides_a);
  const std::vector<int64_t> offsets_b =
      precompute_batch_offsets(num_batches, batch_out, padded_b, strides_b);

  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(N) + block.x - 1) / block.x,
            (static_cast<unsigned>(M) + block.y - 1) / block.y);
  for (int64_t batch = 0; batch < num_batches; ++batch) {
    matmul_2d_kernel<<<grid, block>>>(
        c_a.data<float>() + offsets_a[batch], c_b.data<float>() + offsets_b[batch],
        out.data<float>() + batch * matrix_numel, M, K, N);
  }
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  assert_same_device(a, b);
  return launch_broadcast_binary(a.contiguous(), b.contiguous(), 0);
}

Tensor mul(const Tensor& a, const Tensor& b) {
  assert_same_device(a, b);
  return launch_broadcast_binary(a.contiguous(), b.contiguous(), 1);
}

Tensor neg(const Tensor& t) {
  Tensor c = t.contiguous();
  Tensor out(t.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((t.numel() + block - 1) / block);
  neg_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), t.numel());
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor gelu(const Tensor& t) {
  Tensor c = t.contiguous();
  Tensor out(t.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((t.numel() + block - 1) / block);
  gelu_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), t.numel());
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor relu(const Tensor& t) {
  Tensor c = t.contiguous();
  Tensor out(t.shape(), DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((t.numel() + block - 1) / block);
  relu_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), t.numel());
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor softmax(const Tensor& x) {
  Tensor c = x.contiguous();
  Tensor out(x.shape(), DType::Float32, Device::CUDA);
  const int64_t cols = x.shape().back();
  const int64_t rows = x.numel() / cols;
  softmax_rows_kernel<<<static_cast<int>(rows), 1>>>(
      c.data<float>(), out.data<float>(), rows, cols);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps) {
  assert_same_device(x, gamma);
  assert_same_device(x, beta);
  Tensor c = x.contiguous();
  Tensor out(x.shape(), DType::Float32, Device::CUDA);
  const int64_t cols = x.shape().back();
  const int64_t rows = x.numel() / cols;
  layernorm_kernel<<<static_cast<int>(rows), 1>>>(
      c.data<float>(), gamma.data<float>(), beta.data<float>(),
      out.data<float>(), rows, cols, eps);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor sum(const Tensor& t) {
  Tensor c = t.contiguous();
  Tensor out({1}, DType::Float32, Device::CUDA);
  TIRAMISU_CUDA_CHECK(cudaMemset(out.data<float>(), 0, sizeof(float)));
  const int block = 256;
  const int grid = std::min(256, static_cast<int>((c.numel() + block - 1) / block));
  sum_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), c.numel());
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor mean(const Tensor& t) {
  Tensor s = sum(t);
  Tensor out({1}, DType::Float32, Device::CUDA);
  float host_sum = 0.0f;
  TIRAMISU_CUDA_CHECK(cudaMemcpy(&host_sum, s.data<float>(), sizeof(float),
                                 cudaMemcpyDeviceToHost));
  const float m = host_sum / static_cast<float>(t.numel());
  TIRAMISU_CUDA_CHECK(
      cudaMemcpy(out.data<float>(), &m, sizeof(float), cudaMemcpyHostToDevice));
  return out;
}

Tensor embedding(const Tensor& weight, const Tensor& indices) {
  assert_same_device(weight, indices);
  const int64_t batch = indices.shape()[0];
  const int64_t seq = indices.shape()[1];
  const int64_t dim = weight.shape()[1];
  Tensor c_indices = indices.contiguous();
  Tensor out({batch, seq, dim}, DType::Float32, Device::CUDA);
  const int block = 256;
  const int grid = static_cast<int>((batch * seq + block - 1) / block);
  embedding_kernel<<<grid, block>>>(weight.data<float>(), c_indices.data<float>(),
                                    out.data<float>(), batch, seq, dim);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor merge_heads(const Tensor& x, int64_t d_model) {
  const int64_t batch = x.shape()[0];
  const int64_t heads = x.shape()[1];
  const int64_t seq = x.shape()[2];
  const int64_t d_k = x.shape()[3];
  Tensor c = x.contiguous();
  Tensor out({batch, seq, d_model}, DType::Float32, Device::CUDA);
  const int block = 256;
  const int64_t total = batch * seq * d_model;
  const int grid = static_cast<int>((total + block - 1) / block);
  merge_heads_kernel<<<grid, block>>>(c.data<float>(), out.data<float>(), batch,
                                      heads, seq, d_k, d_model);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace tiramisu::ops::cuda

#endif  // TIRAMISU_CUDA_ENABLED

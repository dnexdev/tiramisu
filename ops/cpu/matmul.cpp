#include "tiramisu/ops/matmul.hpp"

#include <immintrin.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tiramisu/ops/broadcast.hpp"

namespace tiramisu::ops {

namespace {

constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
constexpr int64_t TILE_K = 64;

void matmul_tile(const float* A, const float* B, float* C, int64_t tm,
                 int64_t tn, int64_t tk, int64_t M_stride, int64_t K_stride,
                 int64_t N_stride) {
  for (int64_t ii = 0; ii < tm; ii++) {
    for (int64_t kk = 0; kk < tk; kk++) {
      float a_val = A[ii * M_stride + kk];
      __m256 a_bcast = _mm256_set1_ps(a_val);

      int64_t jj = 0;
      for (; jj + 8 <= tn; jj += 8) {
        __m256 c_vec = _mm256_loadu_ps(&C[ii * N_stride + jj]);
        __m256 b_vec = _mm256_loadu_ps(&B[kk * K_stride + jj]);
        c_vec = _mm256_fmadd_ps(a_bcast, b_vec, c_vec);
        _mm256_storeu_ps(&C[ii * N_stride + jj], c_vec);
      }
      for (; jj < tn; jj++) {
        C[ii * N_stride + jj] += a_val * B[kk * K_stride + jj];
      }
    }
  }
}

void matmul_2d_accumulate(const float* A, const float* B, float* C, int64_t M,
                          int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; i += TILE_M) {
    for (int64_t j = 0; j < N; j += TILE_N) {
      for (int64_t k = 0; k < K; k += TILE_K) {
        int64_t tm = std::min(TILE_M, M - i);
        int64_t tn = std::min(TILE_N, N - j);
        int64_t tk = std::min(TILE_K, K - k);

        matmul_tile(A + i * K + k, B + k * N + j, C + i * N + j, tm, tn, tk,
                    K, N, N);
      }
    }
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
  int64_t batch_rank = static_cast<int64_t>(shape.size()) - 2;
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
      const int64_t idx =
          padded_operand_batch[i] == 1 ? 0 : coords[i];
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

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("matmul: only Float32 supported");
  }

  const auto& shape_a = a.shape();
  const auto& shape_b = b.shape();
  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::invalid_argument("matmul: tensors must be at least 2D");
  }

  int64_t M = shape_a[shape_a.size() - 2];
  int64_t K = shape_a[shape_a.size() - 1];
  int64_t K_b = shape_b[shape_b.size() - 2];
  int64_t N = shape_b[shape_b.size() - 1];
  if (K != K_b) {
    throw std::invalid_argument("matmul: inner dimensions must match");
  }

  std::vector<int64_t> batch_a = batch_dims(shape_a);
  std::vector<int64_t> batch_b = batch_dims(shape_b);
  std::vector<int64_t> batch_out = broadcast_shapes(batch_a, batch_b);

  std::vector<int64_t> out_shape = batch_out;
  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor c_a = a.contiguous();
  Tensor c_b = b.contiguous();
  Tensor out(out_shape, DType::Float32, Device::CPU);

  const float* A = c_a.data<float>();
  const float* B = c_b.data<float>();
  float* C = out.data<float>();
  std::fill_n(C, out.numel(), 0.0f);

  const int64_t num_batches = product(batch_out);
  const int64_t matrix_numel_c = M * N;
  const auto padded_a = pad_batch_shape(batch_a, batch_out.size());
  const auto padded_b = pad_batch_shape(batch_b, batch_out.size());
  const auto strides_a = batch_strides(shape_a);
  const auto strides_b = batch_strides(shape_b);
  const std::vector<int64_t> offsets_a =
      precompute_batch_offsets(num_batches, batch_out, padded_a, strides_a);
  const std::vector<int64_t> offsets_b =
      precompute_batch_offsets(num_batches, batch_out, padded_b, strides_b);

  for (int64_t batch = 0; batch < num_batches; ++batch) {
    matmul_2d_accumulate(A + offsets_a[batch], B + offsets_b[batch],
                         C + batch * matrix_numel_c, M, K, N);
  }

  return out;
}

}  // namespace tiramisu::ops

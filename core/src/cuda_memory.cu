#include "tiramisu/core/cuda_common.hpp"

#include <cuda_runtime.h>

namespace tiramisu::cuda_mem {
namespace detail {

__global__ void fill_kernel(float* dst, float value, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = value;
  }
}

__global__ void add_kernel(float* dst, const float* src, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] += src[i];
  }
}

__global__ void contiguous_kernel(const float* src, float* dst, int64_t numel,
                                  const int64_t* shape, const int64_t* strides,
                                  int64_t rank) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel) {
    return;
  }
  int64_t temp = i;
  int64_t src_idx = 0;
  for (int64_t d = rank - 1; d >= 0; --d) {
    const int64_t coord = temp % shape[d];
    src_idx += coord * strides[d];
    temp /= shape[d];
  }
  dst[i] = src[src_idx];
}

void fill_f32_cuda(float* dst, float value, int64_t count) {
  const int block = 256;
  const int grid = static_cast<int>((count + block - 1) / block);
  fill_kernel<<<grid, block>>>(dst, value, count);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
}

void add_f32_cuda(float* dst, const float* src, int64_t count) {
  const int block = 256;
  const int grid = static_cast<int>((count + block - 1) / block);
  add_kernel<<<grid, block>>>(dst, src, count);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
}

void copy_bytes_cuda(const void* src, void* dst, std::size_t nbytes,
                     cudaMemcpyKind kind) {
  TIRAMISU_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, kind));
}

void contiguous_strided_cuda(const float* src, float* dst, int64_t numel,
                             const int64_t* shape, const int64_t* strides,
                             int64_t rank) {
  int64_t* d_shape = nullptr;
  int64_t* d_strides = nullptr;
  TIRAMISU_CUDA_CHECK(
      cudaMalloc(&d_shape, static_cast<std::size_t>(rank) * sizeof(int64_t)));
  TIRAMISU_CUDA_CHECK(cudaMalloc(&d_strides,
                                 static_cast<std::size_t>(rank) * sizeof(int64_t)));
  TIRAMISU_CUDA_CHECK(cudaMemcpy(d_shape, shape,
                                 static_cast<std::size_t>(rank) * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
  TIRAMISU_CUDA_CHECK(cudaMemcpy(d_strides, strides,
                                 static_cast<std::size_t>(rank) * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
  const int block = 256;
  const int grid = static_cast<int>((numel + block - 1) / block);
  contiguous_kernel<<<grid, block>>>(src, dst, numel, d_shape, d_strides, rank);
  TIRAMISU_CUDA_CHECK(cudaGetLastError());
  cudaFree(d_shape);
  cudaFree(d_strides);
}

}  // namespace detail
}  // namespace tiramisu::cuda_mem

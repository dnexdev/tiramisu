#include "tiramisu/core/cuda_memory.hpp"

#include <cstring>
#include <stdexcept>

#include "tiramisu/core/tensor.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace tiramisu::cuda_mem {

#ifdef TIRAMISU_CUDA_ENABLED
namespace detail {

void fill_f32_cuda(float* dst, float value, int64_t count);
void add_f32_cuda(float* dst, const float* src, int64_t count);
void copy_bytes_cuda(const void* src, void* dst, std::size_t nbytes,
                     cudaMemcpyKind kind);
void contiguous_strided_cuda(const float* src, float* dst, int64_t numel,
                             const int64_t* shape, const int64_t* strides,
                             int64_t rank);

}  // namespace detail
#endif

void copy_bytes(const void* src, void* dst, std::size_t nbytes, Device src_dev,
                Device dst_dev) {
  if (nbytes == 0) {
    return;
  }
  if (src_dev == Device::CPU && dst_dev == Device::CPU) {
    std::memcpy(dst, src, nbytes);
    return;
  }
#ifdef TIRAMISU_CUDA_ENABLED
  cudaMemcpyKind kind;
  if (src_dev == Device::CPU && dst_dev == Device::CUDA) {
    kind = cudaMemcpyHostToDevice;
  } else if (src_dev == Device::CUDA && dst_dev == Device::CPU) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyDeviceToDevice;
  }
  detail::copy_bytes_cuda(src, dst, nbytes, kind);
#else
  throw std::runtime_error("CUDA copy requested but CUDA is not enabled");
#endif
}

void fill_f32(float* dst, float value, int64_t count, Device device) {
  if (count == 0) {
    return;
  }
  if (device == Device::CUDA) {
#ifdef TIRAMISU_CUDA_ENABLED
    detail::fill_f32_cuda(dst, value, count);
#else
    throw std::runtime_error("CUDA fill requested but CUDA is not enabled");
#endif
  } else {
    for (int64_t i = 0; i < count; ++i) {
      dst[i] = value;
    }
  }
}

void add_f32(float* dst, const float* src, int64_t count, Device device) {
  if (count == 0) {
    return;
  }
  if (device == Device::CUDA) {
#ifdef TIRAMISU_CUDA_ENABLED
    detail::add_f32_cuda(dst, src, count);
#else
    throw std::runtime_error("CUDA add requested but CUDA is not enabled");
#endif
  } else {
    for (int64_t i = 0; i < count; ++i) {
      dst[i] += src[i];
    }
  }
}

void contiguous_copy(const Tensor& src, Tensor& dst) {
  if (src.device() != dst.device()) {
    throw std::runtime_error("contiguous_copy: device mismatch");
  }
  if (src.is_contiguous()) {
    copy_bytes(src.data<float>(), dst.data<float>(),
               static_cast<std::size_t>(src.numel()) * sizeof(float),
               src.device(), dst.device());
    return;
  }

  if (src.device() == Device::CUDA) {
#ifdef TIRAMISU_CUDA_ENABLED
    const int64_t rank = static_cast<int64_t>(src.shape().size());
    detail::contiguous_strided_cuda(
        src.data<float>(), dst.data<float>(), src.numel(), src.shape().data(),
        src.strides().data(), rank);
#else
    throw std::runtime_error("CUDA contiguous requested but CUDA is not enabled");
#endif
    return;
  }

  const std::size_t itemsize = sizeof(float);
  const std::byte* src_base =
      reinterpret_cast<const std::byte*>(src.data<float>());
  std::byte* dst_bytes = reinterpret_cast<std::byte*>(dst.data<float>());
  const int64_t total_elements = src.numel();
  const int64_t rank = static_cast<int64_t>(src.shape().size());

  for (int64_t i = 0; i < total_elements; ++i) {
    int64_t temp = i;
    int64_t src_flat_idx = 0;
    for (int64_t d = rank - 1; d >= 0; --d) {
      const int64_t coord = temp % src.shape()[d];
      src_flat_idx += coord * src.strides()[d];
      temp /= src.shape()[d];
    }
    std::memcpy(dst_bytes + i * itemsize, src_base + src_flat_idx * itemsize,
                itemsize);
  }
}

float read_scalar_f32(const Tensor& t) {
  if (t.device() == Device::CPU) {
    return t.data<float>()[0];
  }
#ifdef TIRAMISU_CUDA_ENABLED
  float host = 0.0f;
  detail::copy_bytes_cuda(t.data<float>(), &host, sizeof(float),
                          cudaMemcpyDeviceToHost);
  return host;
#else
  throw std::runtime_error("CUDA scalar read requested but CUDA is not enabled");
#endif
}

}  // namespace tiramisu::cuda_mem

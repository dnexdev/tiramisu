#include "tiramisu/core/storage.hpp"

#include <cassert>
#include <cstdlib>
#include <new>
#include <stdexcept>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#include "tiramisu/core/dtype.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace tiramisu {

Storage::Storage(std::size_t count, DType dtype, Device device,
                 std::size_t alignment)
    : count_(count), dtype_(dtype), device_(device), alignment_(alignment) {
  // Alignment must be a power of two — aligned_alloc / cuda alignment
  // requirements both assume this.
  assert((alignment & (alignment - 1)) == 0);

  // Reject a CUDA request when CUDA is not compiled in *before* doing any
  // size math (which itself is unchecked for overflow, see Tensor::numel).
#ifndef TIRAMISU_CUDA_ENABLED
  if (device == Device::CUDA) {
    throw std::runtime_error("CUDA storage requested but CUDA is not enabled");
  }
#endif

  if (count == 0) {
    data_ = nullptr;
    return;
  }

  std::size_t bytes = nbytes();
  std::size_t alloc_size = (bytes + alignment_ - 1) & ~(alignment_ - 1);

#ifdef TIRAMISU_CUDA_ENABLED
  if (device == Device::CUDA) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, alloc_size);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    data_ = static_cast<std::byte*>(ptr);
    return;
  }
#endif

#ifdef _MSC_VER
  data_ = static_cast<std::byte*>(_aligned_malloc(alloc_size, alignment_));
#else
  data_ = static_cast<std::byte*>(std::aligned_alloc(alignment_, alloc_size));
#endif
  if (!data_) {
    throw std::bad_alloc();
  }
}

Storage::~Storage() {
  if (!data_) {
    return;
  }

#ifdef TIRAMISU_CUDA_ENABLED
  if (device_ == Device::CUDA) {
    cudaFree(data_);
    return;
  }
#endif

#ifdef _MSC_VER
  _aligned_free(data_);
#else
  std::free(data_);
#endif
}

std::byte* Storage::data() { return data_; }

const std::byte* Storage::data() const { return data_; }

std::size_t Storage::numel() const { return count_; }

std::size_t Storage::nbytes() const { return count_ * dtype_size(dtype_); }

DType Storage::dtype() const { return dtype_; }

Device Storage::device() const { return device_; }

}  // namespace tiramisu

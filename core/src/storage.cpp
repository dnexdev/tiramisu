#include "tiramisu/core/storage.hpp"

#include <cassert>
#include <cstdlib>
#include <new>
#include <stdexcept>

#include "tiramisu/core/dtype.hpp"

namespace tiramisu {

Storage::Storage(std::size_t count, DType dtype, Device device,
                 std::size_t alignment)
    : count_(count), dtype_(dtype), device_(device), alignment_(alignment) {
  // alignment should be a power of 2
  assert((alignment & (alignment - 1)) == 0);

  if (count == 0) {
    data_ = nullptr;
    return;
  }

  std::size_t bytes = nbytes();

  // round up to nearest multiple of a power of 2
  std::size_t alloc_size = (bytes + alignment_ - 1) & ~(alignment_ - 1);

  data_ = static_cast<std::byte*>(std::aligned_alloc(alignment_, alloc_size));

  if (!data_) {
    throw std::bad_alloc();
  }
}

Storage::~Storage() {
  if (data_) {
    std::free(data_);
  }
}

std::byte* Storage::data() { return data_; }

const std::byte* Storage::data() const { return data_; }

std::size_t Storage::numel() const { return count_; }

std::size_t Storage::nbytes() const { return count_ * dtype_size(dtype_); }

DType Storage::dtype() const { return dtype_; }

Device Storage::device() const { return device_; }

}  // namespace tiramisu
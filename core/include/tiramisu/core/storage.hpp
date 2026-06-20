#pragma once
#include <cstddef>

#include "tiramisu/core/device.hpp"
#include "tiramisu/core/dtype.hpp"

namespace tiramisu {

class Storage {
  public:
    Storage(std::size_t count, DType dtype, Device device,
            std::size_t alignment = 64);
    ~Storage();

    // Ownership flows through shared_ptr, not copies.
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    std::byte* data();
    const std::byte* data() const;

    std::size_t numel() const; // element count
    std::size_t nbytes() const;
    DType dtype() const;
    Device device() const;
  
  private:
    std::byte* data_ = nullptr;
    std::size_t count_ = 0;
    DType dtype_;
    Device device_;
    std::size_t alignment_;
};

}
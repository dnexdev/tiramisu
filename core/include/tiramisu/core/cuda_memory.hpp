#pragma once

#include <cstddef>
#include <cstdint>

#include "tiramisu/core/device.hpp"

namespace tiramisu {

class Tensor;

namespace cuda_mem {

void copy_bytes(const void* src, void* dst, std::size_t nbytes, Device src_dev,
                Device dst_dev);
void fill_f32(float* dst, float value, int64_t count, Device device);
void add_f32(float* dst, const float* src, int64_t count, Device device);
void contiguous_copy(const Tensor& src, Tensor& dst);
float read_scalar_f32(const Tensor& t);

}  // namespace cuda_mem
}  // namespace tiramisu

#pragma once
#include <cstdint>
#include <cstddef>

namespace tiramisu {

enum class DType {
  Float32,
  Int32
};

std::size_t dtype_size(DType dtype);

template <typename T>
constexpr DType dtype_of();

template <>
constexpr DType dtype_of<float>() {
  return DType::Float32;
}

template <>
constexpr DType dtype_of<int32_t>() {
  return DType::Int32;
}

}
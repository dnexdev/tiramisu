#pragma once

#include <cstdint>
#include <vector>

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::quant {

// Weight-only, per-channel symmetric int8 quantization.
//
//   scale[c] = max(|w[c, :, ...]|) / 127
//   q[c, ...] = clip(round(w[c, ...] / scale[c]), -127, 127)
//
// One fp32 scale per output channel. Zero-point is implicit (symmetric),
// so dequantization is a single per-channel multiply.
struct QuantizedTensor {
  std::vector<int8_t> data;
  std::vector<float> scales;       // length == shape[channel_axis]
  std::vector<int64_t> shape;
  int64_t channel_axis;            // usually 0
};

// Quantize a fp32 contiguous tensor along `channel_axis`. Rank must be >= 2;
// for 2-D weight matrices (out, in) pass `channel_axis = 0`.
QuantizedTensor quantize_per_channel(const Tensor& weight,
                                     int64_t channel_axis = 0);

// Dequantize back to a fresh fp32 Tensor. Uses stored scales; the round-trip
// error is bounded by 1/(2 * 127) of the per-channel range.
Tensor dequantize(const QuantizedTensor& q);

}  // namespace tiramisu::quant

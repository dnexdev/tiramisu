#include "tiramisu/quant/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tiramisu::quant {

namespace {

// For a contiguous tensor with shape [d0, d1, ..., dN] and channel_axis=k,
// the buffer looks like an (outer x channel x inner) 3-D block where:
//   outer   = product of dims before k
//   channel = shape[k]
//   inner   = product of dims after k
// Linear index for (o, c, i): o * channel * inner + c * inner + i.
struct AxisLayout {
  int64_t outer;
  int64_t channel;
  int64_t inner;
};

AxisLayout compute_layout(const std::vector<int64_t>& shape, int64_t axis) {
  AxisLayout out{1, shape[axis], 1};
  for (int64_t i = 0; i < axis; ++i) out.outer *= shape[i];
  for (size_t i = axis + 1; i < shape.size(); ++i) out.inner *= shape[i];
  return out;
}

}  // namespace

QuantizedTensor quantize_per_channel(const Tensor& weight, int64_t channel_axis) {
  if (weight.shape().size() < 2) {
    throw std::invalid_argument("quantize_per_channel: rank must be >= 2");
  }
  if (channel_axis < 0 ||
      channel_axis >= static_cast<int64_t>(weight.shape().size())) {
    throw std::invalid_argument("quantize_per_channel: channel_axis out of range");
  }

  const Tensor w = weight.is_contiguous() ? weight : weight.contiguous();
  const AxisLayout ax = compute_layout(w.shape(), channel_axis);
  const float* src = w.data<float>();

  QuantizedTensor out;
  out.shape = w.shape();
  out.channel_axis = channel_axis;
  out.data.resize(static_cast<size_t>(w.numel()));
  out.scales.assign(static_cast<size_t>(ax.channel), 0.0f);

  // Pass 1: per-channel max abs.
  for (int64_t o = 0; o < ax.outer; ++o) {
    for (int64_t c = 0; c < ax.channel; ++c) {
      const float* row = src + (o * ax.channel + c) * ax.inner;
      float m = 0.0f;
      for (int64_t i = 0; i < ax.inner; ++i) {
        const float a = std::fabs(row[i]);
        if (a > m) m = a;
      }
      if (m > out.scales[static_cast<size_t>(c)]) {
        out.scales[static_cast<size_t>(c)] = m;
      }
    }
  }
  for (float& s : out.scales) {
    // scale = max_abs / 127; guard against all-zero channels.
    s = (s == 0.0f) ? 1.0f : s / 127.0f;
  }

  // Pass 2: quantize.
  for (int64_t o = 0; o < ax.outer; ++o) {
    for (int64_t c = 0; c < ax.channel; ++c) {
      const float inv_scale = 1.0f / out.scales[static_cast<size_t>(c)];
      const float* row = src + (o * ax.channel + c) * ax.inner;
      int8_t* dst = out.data.data() + (o * ax.channel + c) * ax.inner;
      for (int64_t i = 0; i < ax.inner; ++i) {
        int32_t q = static_cast<int32_t>(std::lrintf(row[i] * inv_scale));
        q = std::clamp(q, -127, 127);
        dst[i] = static_cast<int8_t>(q);
      }
    }
  }

  return out;
}

Tensor dequantize(const QuantizedTensor& q) {
  Tensor out(q.shape);
  const AxisLayout ax = compute_layout(q.shape, q.channel_axis);
  float* dst = out.data<float>();
  const int8_t* src = q.data.data();

  for (int64_t o = 0; o < ax.outer; ++o) {
    for (int64_t c = 0; c < ax.channel; ++c) {
      const float scale = q.scales[static_cast<size_t>(c)];
      const int8_t* row = src + (o * ax.channel + c) * ax.inner;
      float* out_row = dst + (o * ax.channel + c) * ax.inner;
      for (int64_t i = 0; i < ax.inner; ++i) {
        out_row[i] = static_cast<float>(row[i]) * scale;
      }
    }
  }

  return out;
}

}  // namespace tiramisu::quant

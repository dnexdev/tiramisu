// quantize_checkpoint: fp32 v1 checkpoint -> int8 v2 checkpoint.
//
// Weight-only, per-channel symmetric int8. Every parameter with rank >= 2
// (i.e. Linear/Embedding weights) is quantized on `axis=0`; rank-1 params
// (biases, LayerNorm gamma/beta) stay fp32 — they're negligible size and
// quality-critical.
//
// Usage:
//   quantize_checkpoint in.ckpt out.ckpt

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/quant/quantize.hpp"
#include "tiramisu/serialize/checkpoint.hpp"

using namespace tiramisu;
using namespace tiramisu::serialize;

namespace {

int64_t byte_size_of(const RawParameter& p) {
  int64_t numel = 1;
  for (int64_t d : p.shape) numel *= d;
  return p.kind == kInt8PerChannelSymmetric
             ? numel + static_cast<int64_t>(p.scales.size() * sizeof(float))
             : numel * static_cast<int64_t>(sizeof(float));
}

RawParameter quantize_entry(const ParameterEntry& in) {
  RawParameter out;
  out.name = in.name;
  out.shape = in.shape;

  if (in.shape.size() < 2) {
    out.kind = kFp32;
    out.fp32_data = in.data;
    return out;
  }

  Tensor t(in.shape);
  std::memcpy(t.data<float>(), in.data.data(), in.data.size() * sizeof(float));
  quant::QuantizedTensor q = quant::quantize_per_channel(t, 0);
  out.kind = kInt8PerChannelSymmetric;
  out.channel_axis = 0;
  out.scales = std::move(q.scales);
  out.int8_data = std::move(q.data);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s in.ckpt out.ckpt\n", argv[0]);
    return 1;
  }

  try {
    GPTCheckpoint in = load_gpt_checkpoint(argv[1]);

    RawGPTCheckpoint out;
    out.config = in.config;
    out.step = in.step;
    out.epoch = in.epoch;
    out.parameters.reserve(in.parameters.size());

    int64_t fp32_bytes = 0, out_bytes = 0;
    for (const ParameterEntry& p : in.parameters) {
      fp32_bytes += static_cast<int64_t>(p.data.size() * sizeof(float));
      RawParameter r = quantize_entry(p);
      out_bytes += byte_size_of(r);
      out.parameters.push_back(std::move(r));
    }

    save_raw_gpt_checkpoint(argv[2], out);
    std::printf("wrote %s\n  fp32 weights: %.2f MB\n  int8 output:  %.2f MB "
                "(ratio %.2fx)\n",
                argv[2], fp32_bytes / 1e6, out_bytes / 1e6,
                fp32_bytes / static_cast<double>(std::max<int64_t>(out_bytes, 1)));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "quantize_checkpoint: %s\n", e.what());
    return 1;
  }
}

#include "tiramisu/nn/kv_cache.hpp"

#include <cstring>
#include <stdexcept>

#include "tiramisu/core/cuda_memory.hpp"

namespace tiramisu::nn {

namespace {

void copy_kv_block(const Tensor& src, Tensor& dst, int64_t src_seq, int64_t dst_seq) {
  const int64_t batch = src.shape()[0];
  const int64_t heads = src.shape()[1];
  const int64_t d_k = src.shape()[3];
  Tensor src_c = src.contiguous();
  Tensor dst_c = dst.contiguous();

  if (src.device() != Device::CPU || dst.device() != Device::CPU) {
    throw std::runtime_error("kv_cache: CPU-only for inference cache");
  }

  const float* src_data = src_c.data<float>();
  float* dst_data = dst_c.data<float>();
  const int64_t src_stride = src.shape()[2];
  const int64_t dst_stride = dst.shape()[2];

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      const std::size_t src_off = static_cast<std::size_t>(
          (((b * heads + h) * src_stride) + src_seq) * d_k);
      const std::size_t dst_off = static_cast<std::size_t>(
          (((b * heads + h) * dst_stride) + dst_seq) * d_k);
      std::memcpy(dst_data + dst_off, src_data + src_off,
                  static_cast<std::size_t>(d_k) * sizeof(float));
    }
  }
}

}  // namespace

void GPTKVCache::reset() {
  layers.clear();
  seq_len = 0;
}

Tensor kv_cache_append(const Tensor& cache, const Tensor& token_kv) {
  if (cache.shape().size() != 4 || token_kv.shape().size() != 4) {
    throw std::invalid_argument("kv_cache_append: expected 4D tensors");
  }
  const int64_t batch = cache.shape()[0];
  const int64_t heads = cache.shape()[1];
  const int64_t old_seq = cache.shape()[2];
  const int64_t d_k = cache.shape()[3];
  if (token_kv.shape() != std::vector<int64_t>({batch, heads, 1, d_k})) {
    throw std::invalid_argument("kv_cache_append: token_kv shape mismatch");
  }

  Tensor out({batch, heads, old_seq + 1, d_k}, DType::Float32, cache.device());
  for (int64_t s = 0; s < old_seq; ++s) {
    copy_kv_block(cache, out, s, s);
  }
  copy_kv_block(token_kv, out, 0, old_seq);
  return out;
}

Tensor kv_cache_trim(const Tensor& cache) {
  if (cache.shape()[2] <= 1) {
    throw std::invalid_argument("kv_cache_trim: cache too short to trim");
  }
  const int64_t batch = cache.shape()[0];
  const int64_t heads = cache.shape()[1];
  const int64_t old_seq = cache.shape()[2];
  const int64_t d_k = cache.shape()[3];
  Tensor out({batch, heads, old_seq - 1, d_k}, DType::Float32, cache.device());
  for (int64_t s = 1; s < old_seq; ++s) {
    copy_kv_block(cache, out, s, s - 1);
  }
  return out;
}

}  // namespace tiramisu::nn

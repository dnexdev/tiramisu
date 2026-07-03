#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::nn {

struct KVCacheLayer {
  std::optional<Tensor> k;  // [batch, num_heads, seq_len, d_k]
  std::optional<Tensor> v;  // [batch, num_heads, seq_len, d_k]
};

struct GPTKVCache {
  std::vector<KVCacheLayer> layers;
  int64_t seq_len = 0;

  void reset();
};

// Append [batch, heads, 1, d_k] tensors along the sequence dimension.
Tensor kv_cache_append(const Tensor& cache, const Tensor& token_kv);

// Drop the oldest sequence position (dim 2).
Tensor kv_cache_trim(const Tensor& cache);

}  // namespace tiramisu::nn

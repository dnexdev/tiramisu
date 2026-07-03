#pragma once

#include "tiramisu/nn/kv_cache.hpp"
#include "tiramisu/nn/feed_forward.hpp"
#include "tiramisu/nn/layernorm.hpp"
#include "tiramisu/nn/module.hpp"
#include "tiramisu/nn/multi_head_attention.hpp"

namespace tiramisu::nn {

class TransformerBlock : public Module {
 public:
  TransformerBlock(int64_t d_model, int64_t num_heads,
                   Device device = Device::CPU);

  Tensor forward(const Tensor& x) override;
  Tensor forward_prefill(const Tensor& x, KVCacheLayer& cache);
  Tensor forward_decode(const Tensor& x, KVCacheLayer& cache);
  std::vector<Tensor*> parameters() override;

 private:
  LayerNorm ln1_;
  MultiHeadAttention mha_;
  LayerNorm ln2_;
  FeedForward ffn_;
};

}  // namespace tiramisu::nn

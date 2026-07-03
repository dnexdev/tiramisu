#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tiramisu/nn/embedding.hpp"
#include "tiramisu/nn/kv_cache.hpp"
#include "tiramisu/nn/layernorm.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/module.hpp"
#include "tiramisu/nn/transformer_block.hpp"

namespace tiramisu::nn {

struct GPTConfig {
  int64_t vocab_size;
  int64_t d_model;
  int64_t num_heads;
  int64_t num_layers;
  int64_t max_seq_len;
  bool tie_weights = false;
};

class GPT : public Module {
 public:
  explicit GPT(const GPTConfig& config, Device device = Device::CPU);

  Tensor forward(const Tensor& token_ids) override;
  Tensor prefill(const Tensor& token_ids, GPTKVCache& cache);
  Tensor decode_step(int64_t token_id, GPTKVCache& cache);
  std::vector<Tensor*> parameters() override;

  const GPTConfig& config() const { return config_; }
  int64_t count_parameters();

 private:
  GPTConfig config_;
  Embedding tok_emb_;
  Embedding pos_emb_;
  std::vector<std::shared_ptr<TransformerBlock>> blocks_;
  LayerNorm ln_f_;
  Linear lm_head_;

  Tensor logits_from_hidden(const Tensor& hidden_last);
  Tensor embed_tokens(const Tensor& token_ids, int64_t start_pos);
  Tensor embed_single_token(int64_t token_id, int64_t pos, Device device);
};

}  // namespace tiramisu::nn

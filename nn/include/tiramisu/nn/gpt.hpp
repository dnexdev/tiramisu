#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tiramisu/nn/embedding.hpp"
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
};

class GPT : public Module {
 public:
  explicit GPT(const GPTConfig& config);

  Tensor forward(const Tensor& token_ids) override;
  std::vector<Tensor*> parameters() override;

  const GPTConfig& config() const { return config_; }

 private:
  GPTConfig config_;
  Embedding tok_emb_;
  Embedding pos_emb_;
  std::vector<std::shared_ptr<TransformerBlock>> blocks_;
  LayerNorm ln_f_;
  Linear lm_head_;
};

}  // namespace tiramisu::nn

#include "tiramisu/nn/gpt.hpp"

#include <stdexcept>

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

namespace {

void append_params(std::vector<Tensor*>& dst, Module& module) {
  auto params = module.parameters();
  dst.insert(dst.end(), params.begin(), params.end());
}

}  // namespace

GPT::GPT(const GPTConfig& config)
    : config_(config),
      tok_emb_(config.vocab_size, config.d_model),
      pos_emb_(config.max_seq_len, config.d_model),
      ln_f_(config.d_model),
      lm_head_(config.d_model, config.vocab_size) {
  if (config.d_model % config.num_heads != 0) {
    throw std::invalid_argument("GPT: d_model must be divisible by num_heads");
  }

  blocks_.reserve(config.num_layers);
  for (int64_t i = 0; i < config.num_layers; ++i) {
    blocks_.push_back(
        std::make_shared<TransformerBlock>(config.d_model, config.num_heads));
  }
}

Tensor GPT::forward(const Tensor& token_ids) {
  const int64_t batch = token_ids.shape()[0];
  const int64_t seq = token_ids.shape()[1];
  if (seq > config_.max_seq_len) {
    throw std::invalid_argument("GPT: sequence length exceeds max_seq_len");
  }

  Tensor pos_ids({batch, seq});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      pos_ids.at<float>({b, s}) = static_cast<float>(s);
    }
  }

  Tensor x =
      tiramisu::autograd::add(tok_emb_.forward(token_ids), pos_emb_.forward(pos_ids));
  for (const auto& block : blocks_) {
    x = block->forward(x);
  }
  x = ln_f_.forward(x);
  return lm_head_.forward(x);
}

std::vector<Tensor*> GPT::parameters() {
  std::vector<Tensor*> params;
  append_params(params, tok_emb_);
  append_params(params, pos_emb_);
  for (const auto& block : blocks_) {
    append_params(params, *block);
  }
  append_params(params, ln_f_);
  append_params(params, lm_head_);
  return params;
}

}  // namespace tiramisu::nn

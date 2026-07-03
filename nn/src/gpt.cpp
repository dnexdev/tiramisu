#include "tiramisu/nn/gpt.hpp"

#include <stdexcept>
#include <vector>

#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/cuda_memory.hpp"

namespace tiramisu::nn {

namespace {

void append_params(std::vector<Tensor*>& dst, Module& module) {
  auto params = module.parameters();
  dst.insert(dst.end(), params.begin(), params.end());
}

Tensor make_pos_ids(int64_t batch, int64_t seq, Device device) {
  std::vector<float> host(static_cast<size_t>(batch * seq));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      host[static_cast<size_t>(b * seq + s)] = static_cast<float>(s);
    }
  }
  Tensor pos_ids({batch, seq}, DType::Float32, device);
  cuda_mem::copy_bytes(host.data(), pos_ids.data<float>(),
                       host.size() * sizeof(float), Device::CPU, device);
  return pos_ids;
}

}  // namespace

GPT::GPT(const GPTConfig& config, Device device)
    : config_(config),
      tok_emb_(config.vocab_size, config.d_model, device),
      pos_emb_(config.max_seq_len, config.d_model, device),
      ln_f_(config.d_model, 1e-5f, device),
      lm_head_(config.d_model, config.vocab_size, device) {
  if (config.d_model % config.num_heads != 0) {
    throw std::invalid_argument("GPT: d_model must be divisible by num_heads");
  }

  blocks_.reserve(config.num_layers);
  for (int64_t i = 0; i < config.num_layers; ++i) {
    blocks_.push_back(
        std::make_shared<TransformerBlock>(config.d_model, config.num_heads,
                                           device));
  }
}

Tensor GPT::forward(const Tensor& token_ids) {
  const int64_t batch = token_ids.shape()[0];
  const int64_t seq = token_ids.shape()[1];
  if (seq > config_.max_seq_len) {
    throw std::invalid_argument("GPT: sequence length exceeds max_seq_len");
  }

  Tensor pos_ids = make_pos_ids(batch, seq, token_ids.device());

  Tensor x =
      tiramisu::autograd::add(tok_emb_.forward(token_ids), pos_emb_.forward(pos_ids));
  for (const auto& block : blocks_) {
    x = block->forward(x);
  }
  x = ln_f_.forward(x);

  if (config_.tie_weights) {
    Tensor tied =
        tiramisu::autograd::transpose(tok_emb_.weight(), 0, 1);
    return tiramisu::autograd::add(tiramisu::autograd::matmul(x, tied),
                                   lm_head_.bias());
  }
  return lm_head_.forward(x);
}

Tensor GPT::logits_from_hidden(const Tensor& hidden_last) {
  if (config_.tie_weights) {
    Tensor tied =
        tiramisu::autograd::transpose(tok_emb_.weight(), 0, 1);
    return tiramisu::autograd::add(tiramisu::autograd::matmul(hidden_last, tied),
                                   lm_head_.bias());
  }
  return lm_head_.forward(hidden_last);
}

Tensor GPT::embed_tokens(const Tensor& token_ids, int64_t start_pos) {
  const int64_t batch = token_ids.shape()[0];
  const int64_t seq = token_ids.shape()[1];
  const Device device = token_ids.device();

  std::vector<float> pos_host(static_cast<size_t>(batch * seq));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      pos_host[static_cast<size_t>(b * seq + s)] =
          static_cast<float>(start_pos + s);
    }
  }
  Tensor pos_ids({batch, seq}, DType::Float32, device);
  cuda_mem::copy_bytes(pos_host.data(), pos_ids.data<float>(),
                       pos_host.size() * sizeof(float), Device::CPU, device);
  return tiramisu::autograd::add(tok_emb_.forward(token_ids),
                                 pos_emb_.forward(pos_ids));
}

Tensor GPT::embed_single_token(int64_t token_id, int64_t pos, Device device) {
  Tensor ids({1, 1}, DType::Float32, device);
  const float id_f = static_cast<float>(token_id);
  cuda_mem::copy_bytes(&id_f, ids.data<float>(), sizeof(float), Device::CPU,
                       device);

  std::vector<float> pos_host = {static_cast<float>(pos)};
  Tensor pos_ids({1, 1}, DType::Float32, device);
  cuda_mem::copy_bytes(pos_host.data(), pos_ids.data<float>(),
                       pos_host.size() * sizeof(float), Device::CPU, device);
  return tiramisu::autograd::add(tok_emb_.forward(ids), pos_emb_.forward(pos_ids));
}

Tensor GPT::prefill(const Tensor& token_ids, GPTKVCache& cache) {
  const int64_t seq = token_ids.shape()[1];
  if (seq > config_.max_seq_len) {
    throw std::invalid_argument("GPT::prefill: sequence length exceeds max_seq_len");
  }
  if (cache.layers.size() != static_cast<size_t>(config_.num_layers)) {
    cache.layers.clear();
    cache.layers.resize(static_cast<size_t>(config_.num_layers));
  }

  Tensor x = embed_tokens(token_ids, 0);
  for (int64_t i = 0; i < config_.num_layers; ++i) {
    x = blocks_[static_cast<size_t>(i)]->forward_prefill(x, cache.layers[static_cast<size_t>(i)]);
  }
  x = ln_f_.forward(x);
  cache.seq_len = seq;

  const int64_t d_model = config_.d_model;
  Tensor flat = x.reshape({seq, d_model});
  Tensor hidden_last = flat.slice(0, seq - 1, seq).reshape({1, 1, d_model});
  return logits_from_hidden(hidden_last);
}

Tensor GPT::decode_step(int64_t token_id, GPTKVCache& cache) {
  const Device device = tok_emb_.weight().device();
  const int64_t max_seq = config_.max_seq_len;

  if (cache.seq_len >= max_seq) {
    throw std::invalid_argument(
        "GPT::decode_step: cache full; re-prefill the context instead");
  }

  const int64_t pos = cache.seq_len;
  Tensor x = embed_single_token(token_id, pos, device);
  for (int64_t i = 0; i < config_.num_layers; ++i) {
    x = blocks_[static_cast<size_t>(i)]->forward_decode(x, cache.layers[static_cast<size_t>(i)]);
  }
  x = ln_f_.forward(x);
  cache.seq_len++;

  return logits_from_hidden(x);
}

std::vector<Tensor*> GPT::parameters() {
  std::vector<Tensor*> params;
  append_params(params, tok_emb_);
  append_params(params, pos_emb_);
  for (const auto& block : blocks_) {
    append_params(params, *block);
  }
  append_params(params, ln_f_);
  if (config_.tie_weights) {
    params.push_back(&lm_head_.bias());
  } else {
    append_params(params, lm_head_);
  }
  return params;
}

int64_t GPT::count_parameters() {
  int64_t total = 0;
  for (Tensor* p : parameters()) {
    total += p->numel();
  }
  return total;
}

}  // namespace tiramisu::nn

#include "tiramisu/nn/multi_head_attention.hpp"

#include <cmath>
#include <stdexcept>

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

namespace {

Tensor split_heads(const Tensor& x, int64_t num_heads) {
  const int64_t batch = x.shape()[0];
  const int64_t seq = x.shape()[1];
  const int64_t d_model = x.shape()[2];
  const int64_t d_k = d_model / num_heads;
  return tiramisu::autograd::permute(
      tiramisu::autograd::reshape(x, {batch, seq, num_heads, d_k}),
      {0, 2, 1, 3});
}

Tensor make_causal_mask(int64_t seq_len) {
  Tensor mask({seq_len, seq_len});
  for (int64_t i = 0; i < seq_len; i++) {
    for (int64_t j = 0; j < seq_len; j++) {
      mask.at<float>({i, j}) = (j > i) ? -1e9f : 0.0f;
    }
  }
  return mask;
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(int64_t d_model, int64_t num_heads,
                                       bool causal)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_k_(d_model / num_heads),
      causal_(causal),
      w_q_(d_model, d_model),
      w_k_(d_model, d_model),
      w_v_(d_model, d_model),
      w_o_(d_model, d_model) {
  if (d_model % num_heads != 0) {
    throw std::invalid_argument(
        "MultiHeadAttention: d_model must be divisible by num_heads");
  }
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
  const int64_t seq = x.shape()[1];

  Tensor q = split_heads(w_q_.forward(x), num_heads_);
  Tensor k = split_heads(w_k_.forward(x), num_heads_);
  Tensor v = split_heads(w_v_.forward(x), num_heads_);

  Tensor scores = tiramisu::autograd::matmul(
      q, tiramisu::autograd::transpose(k, -2, -1));

  Tensor scale({1});
  scale.at<float>({0}) =
      1.0f / std::sqrt(static_cast<float>(d_k_));
  scores = tiramisu::autograd::mul(scores, scale);

  if (causal_) {
    if (cached_mask_seq_ != seq) {
      cached_mask_ = make_causal_mask(seq);
      cached_mask_seq_ = seq;
    }
    scores = tiramisu::autograd::add(scores, *cached_mask_);
  }

  Tensor weights = tiramisu::autograd::softmax(scores);
  Tensor context = tiramisu::autograd::matmul(weights, v);
  Tensor merged = tiramisu::autograd::merge_heads(context, d_model_);
  return w_o_.forward(merged);
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
  std::vector<Tensor*> params;
  for (Module* layer :
       {static_cast<Module*>(&w_q_), static_cast<Module*>(&w_k_),
        static_cast<Module*>(&w_v_), static_cast<Module*>(&w_o_)}) {
    auto layer_params = layer->parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
  return params;
}

}  // namespace tiramisu::nn

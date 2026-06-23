#pragma once

#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/module.hpp"

namespace tiramisu::nn {

class MultiHeadAttention : public Module {
 public:
  MultiHeadAttention(int64_t d_model, int64_t num_heads, bool causal = true);

  Tensor forward(const Tensor& x) override;
  std::vector<Tensor*> parameters() override;

 private:
  int64_t d_model_;
  int64_t num_heads_;
  int64_t d_k_;
  bool causal_;

  Linear w_q_;
  Linear w_k_;
  Linear w_v_;
  Linear w_o_;
};

}  // namespace tiramisu::nn

#pragma once

#include <cstdint>
#include <optional>

#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/module.hpp"

namespace tiramisu::nn {

class MultiHeadAttention : public Module {
 public:
  MultiHeadAttention(int64_t d_model, int64_t num_heads, bool causal = true,
                     Device device = Device::CPU);

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

  mutable std::optional<Tensor> cached_mask_;
  mutable int64_t cached_mask_seq_ = -1;
};

}  // namespace tiramisu::nn

#pragma once

#include "tiramisu/nn/module.hpp"
#include "tiramisu/nn/parameter.hpp"

namespace tiramisu::nn {

class Embedding : public Module {
 public:
  Embedding(int64_t vocab_size, int64_t d_model);

  Tensor forward(const Tensor& token_ids) override;
  std::vector<Tensor*> parameters() override;

  Parameter& weight() { return weight_; }
  const Parameter& weight() const { return weight_; }

 private:
  Parameter weight_;
};

}  // namespace tiramisu::nn

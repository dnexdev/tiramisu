#pragma once
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/module.hpp"
#include "tiramisu/nn/parameter.hpp"

namespace tiramisu::nn {

class Linear : public Module {
 public:
  Linear(int64_t in_features, int64_t out_features);

  Tensor forward(const Tensor& x) override;
  std::vector<Tensor*> parameters() override;

  Parameter& weight() { return weight_; }
  Parameter& bias() { return bias_; }

 private:
  Parameter weight_;
  Parameter bias_;
};

}  // namespace tiramisu::nn
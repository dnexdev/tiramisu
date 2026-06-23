#pragma once

#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/module.hpp"

namespace tiramisu::nn {

class FeedForward : public Module {
 public:
  explicit FeedForward(int64_t d_model, Device device = Device::CPU);

  Tensor forward(const Tensor& x) override;
  std::vector<Tensor*> parameters() override;

 private:
  Linear fc1_;
  Linear fc2_;
};

}  // namespace tiramisu::nn

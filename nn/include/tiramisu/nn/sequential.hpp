#pragma once
#include <vector>

#include "tiramisu/nn/module.hpp"

namespace tiramisu::nn {

class Sequential : public Module {
 public:
  Sequential(const std::vector<std::shared_ptr<Module>>& modules);

  Tensor forward(const Tensor& x) override;
  std::vector<Tensor*> parameters() override;

 private:
  std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace tiramisu::nn
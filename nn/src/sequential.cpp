#include "tiramisu/nn/sequential.hpp"

namespace tiramisu::nn {

Sequential::Sequential(const std::vector<std::shared_ptr<Module>>& modules)
  : modules_(modules) {}

Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;
  for (const auto& mod : modules_) {
    out = mod->forward(out);
  }
  return out;
}

std::vector<Tensor*> Sequential::parameters() {
  std::vector<Tensor*> params;
  for (const auto& mod : modules_) {
    auto mod_params = mod->parameters();
    params.insert(params.end(), mod_params.begin(), mod_params.end());
  }
  return params;
}

}
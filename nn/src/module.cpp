#include "tiramisu/nn/module.hpp"

namespace tiramisu::nn {

Module::~Module() = default;

std::vector<Tensor*> Module::parameters() { return {}; }

}  // namespace tiramisu::nn
#pragma once
#include <vector>
#include <memory>
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::nn {

class Module {
  public:
    virtual ~Module();

    virtual Tensor forward(const Tensor& x) = 0;

    virtual std::vector<Tensor*> parameters();
};

}
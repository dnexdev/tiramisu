#pragma once
#include <vector>
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::optim {

class SGD {
  public:
    SGD(const std::vector<Tensor*>& parameters, float lr = 0.01f);

    void step();
    void zero_grad();

  private:
    std::vector<Tensor*> parameters_;
    float lr_;
};
}
#pragma once
#include "tiramisu/nn/module.hpp"
#include "tiramisu/nn/parameter.hpp"

namespace tiramisu::nn {

class LayerNorm : public Module {
  public:
    explicit LayerNorm(int64_t features, float eps = 1e-5f);

    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;
  
  private:
    Parameter gamma_;
    Parameter beta_;
    float eps_;
};

}